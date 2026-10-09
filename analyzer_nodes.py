import os
import base64
import re
import httpx
import os
from validator import ImagingResult, EhrResult, RagResult, HPI
from note_generator import generate_soap_note
from vector_store import vector_store


def data_url_to_bytes(data_url: str) -> bytes | None:
    match = re.match(r"^data:(.+?);base64,(.+)$", data_url)
    if not match:
        return None
    return base64.b64decode(match.group(2))


async def chest_vision_node(state: dict) -> dict:
    hpi: HPI | None = state.get("hpi")
    xray_url = state.get("xrayImageUrl")

    if not hpi or not hpi.cardioRespRelevant:
        return {}
    if not xray_url:
        return {}

    try:
        img_bytes = data_url_to_bytes(xray_url)
        if not img_bytes:
            return {
                "imaging": ImagingResult(
                    model="ViT-B/16 (NIH ChestX-ray14)",
                    labels=[],
                    topFindings=[],
                    note="Uploaded image could not be decoded. Your X-ray was NOT assessed."
                )
            }

        headers = {}
        if os.environ.get("INFERENCE_API_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ.get('INFERENCE_API_TOKEN')}"

        endpoint = os.environ.get(
            "CHESTVISION_ENDPOINT", "") + "/predict"

        async with httpx.AsyncClient() as client:
            files = {"file": ("xray.jpg", img_bytes, "image/jpeg")}
            res = await client.post(endpoint, headers=headers, files=files)

        if res.status_code != 200:
            return {
                "imaging": ImagingResult(
                    model="ViT-B/16 (NIH ChestX-ray14)",
                    labels=[],
                    topFindings=[],
                    note=f"Imaging model unavailable ({res.status_code}). Your X-ray was NOT assessed — this is not a negative result."
                )
            }

        raw = res.json()
        print(f"[HealthPilot] ChestVision raw response: {str(raw)[:1500]}")

        predictions = raw.get("predictions", [])
        if not predictions:
            pass  # Handle unrecognised shape error branch

        labels = [{"pathology": p["label"], "probability": p["probability"]}
                  for p in predictions]
        labels.sort(key=lambda x: x["probability"], reverse=True)

        top_findings = [l["pathology"] for l in labels[:3]]

        return {
            "imaging": ImagingResult(
                model="ViT-B/16 (NIH ChestX-ray14, 14 labels)",
                labels=labels[:5],
                topFindings=top_findings,
                note="Nothing exceeded the tuned decision threshold. This does NOT exclude disease — recall on rare labels is known to be poor." if not top_findings else "Screening signals only. Not a radiological report."
            )
        }
    except Exception as err:
        print(f"[HealthPilot] chestVisionNode failed: {err}")
        return {
            "imaging": ImagingResult(
                model="ViT-B/16 (NIH ChestX-ray14)",
                labels=[],
                topFindings=[],
                note="Imaging model could not be reached. Your X-ray was NOT assessed — this is not a negative result."
            )
        }


def build_tabular_features(hpi: HPI) -> dict:
    p = hpi.patientProfile
    feats = {}

    if p.ageYears is not None:
        feats["AGE"] = p.ageYears
    if p.sex is not None:
        feats["IS_FEMALE"] = 1 if p.sex == "female" else 0

    vitals = {}
    if p.bmi is not None:
        vitals["bmi"] = p.bmi
    if p.systolicBP is not None:
        vitals["systolicBP"] = p.systolicBP
    if p.hba1c is not None:
        vitals["hba1c"] = p.hba1c
    if p.totalCholesterol is not None:
        vitals["totalCholesterol"] = p.totalCholesterol
    if vitals:
        feats["_vitals"] = vitals

    feats["_medications"] = hpi.medications
    feats["_allergies"] = hpi.allergies

    if p.race is not None:
        feats["_race"] = p.race
    if p.ethnicity is not None:
        feats["_ethnicity"] = p.ethnicity

    return feats


async def ehr_node(state: dict) -> dict:
    hpi: HPI | None = state.get("hpi")
    if not hpi:
        return {}

    note = await generate_soap_note(hpi)
    features = build_tabular_features(hpi)

    try:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("INFERENCE_API_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ.get('INFERENCE_API_TOKEN')}"

        endpoint = os.environ.get("EHR_ENDPOINT", "") + "/predict"

        async with httpx.AsyncClient() as client:
            res = await client.post(endpoint, headers=headers, json={
                "features": features,
                "note": note,
                "top_k": 8
            })

        if res.status_code != 200:
            note_msg = f"EHR model unavailable ({res.status_code}). Check EHR_ENDPOINT points at the *.hf.space root."
            if res.status_code == 401:
                note_msg = "EHR model auth failed (401). Check INFERENCE_API_TOKEN matches the Space's INFERENCE_API_TOKEN."
            elif res.status_code == 503:
                note_msg = "EHR model still warming up or busy (503). Assessment proceeds without it."

            return {
                "ehr": EhrResult(
                    model="Hybrid MLP + BioClinical-ModernBERT-large",
                    riskScores=[],
                    medicationFlags=[],
                    note=note_msg
                )
            }

        raw = res.json()
        conditions = raw.get("conditions", [])

        print("[HealthPilot] Hybrid EHR model predictions:\n" +
              ("\n\n".join([f"• {c['label']}\n  Probability: {c['probability']*100:.1f}%\n  Threshold: {c['threshold']*100:.1f}%\n  Predicted: {'✅ Positive' if c['predicted'] == 1 else '❌ Negative'}" for c in conditions]) if conditions else "No conditions returned."))

        flagged = [c for c in conditions if c.get("predicted") == 1]
        chosen = flagged if flagged else conditions[:5]

        return {
            "ehr": EhrResult(
                model="Hybrid MLP + BioClinical-ModernBERT-large",
                riskScores=[{"outcome": c["label"],
                             "probability": c["probability"]} for c in chosen],
                medicationFlags=[],
                note="Conditions flagged above their individually-tuned decision thresholds. Risk priors, not a diagnosis." if flagged else "No condition crossed its tuned threshold; showing strongest signals only. Not a diagnosis."
            )
        }
    except Exception as err:
        print(f"[HealthPilot] ehrNode failed: {err}")
        return {
            "ehr": EhrResult(
                model="Hybrid MLP + BioClinical-ModernBERT-large",
                riskScores=[],
                medicationFlags=[],
                note="EHR model could not be reached. Assessment proceeds without it."
            )
        }


async def rag_node(state: dict) -> dict:
    hpi = state.get("hpi")
    if not hpi:
        return {}

    query = ", ".join([hpi.chiefComplaint] +
                      [s.name for s in hpi.symptoms] + hpi.candidateConditions)

    try:
        # Perform similarity search in Cosmos DB
        results = vector_store.similarity_search_with_score(query, k=6)

        passages = []
        for doc, score in results:
            passages.append({
                "title": doc.metadata.get("title", ""),
                "snippet": doc.page_content[:600],
                "score": score,
                "source": doc.metadata.get("source", "medical-encyclopaedia")
            })

        return {
            "rag": RagResult(query=query, passages=passages)
        }
    except Exception as err:
        print(f"[HealthPilot] ragNode failed: {err}")
        return {"rag": RagResult(query=query, passages=[])}
