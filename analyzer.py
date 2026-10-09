from validator import ImagingResult, EhrResult, RagResult, HPI, Analysis
from llm import medgemma_complete, parse_json_block


def build_evidence_bundle(state: dict) -> str:
    s = []
    hpi: HPI | None = state.get("hpi")

    s.append(
        f"## STRUCTURED HISTORY OF PRESENT ILLNESS\n{hpi.model_dump_json(indent=2) if hpi else 'None'}")

    imaging: ImagingResult | None = state.get("imaging")
    if imaging:
        labels_str = ", ".join(
            [f"{l.pathology} {l.probability*100:.1f}%" for l in imaging.labels])
        top_str = ", ".join(imaging.topFindings) or "none"
        s.append(
            f"## CHEST X-RAY CLASSIFIER ({imaging.model})\nTop probabilities: {labels_str}\nTop 3: {top_str}\nCaveat: {imaging.note}")
    else:
        s.append("## CHEST X-RAY\nNot performed. Absence of imaging is NOT a negative finding. Do not treat it as reassurance.")

    ehr: EhrResult | None = state.get("ehr")
    if ehr:
        scores_str = ", ".join(
            [f"{r.outcome} {r.probability*100:.1f}%" for r in ehr.riskScores])
        flags_str = ", ".join(ehr.medicationFlags) or "none"
        s.append(
            f"## EHR / MEDICATION RISK MODEL ({ehr.model})\nRisk scores: {scores_str}\nMedication flags: {flags_str}\nCaveat: {ehr.note}")
    else:
        s.append("## EHR MODEL\nNo structured record available.")

    rag: RagResult | None = state.get("rag")
    if rag and rag.passages:
        passages_str = "\n\n".join(
            [f"[{i+1}] {p.title} ({p.source})\n{p.snippet}" for i, p in enumerate(rag.passages)])
        s.append(f"## RETRIEVED REFERENCE MATERIAL\n{passages_str}")
    else:
        s.append("## REFERENCE MATERIAL\nNo relevant passages retrieved.")

    s.append("## TASK\nSynthesise the above. Where the model outputs and the patient's history disagree, say so explicitly. Return the JSON object only.")

    return "\n\n".join(s)


async def analyser_node(state: dict) -> dict:
    user = build_evidence_bundle(state)

    print(
        f"\n  ==================== ANALYSER INPUT ====================\n\n  {user}\n\n  ==============================================================\n  ")

    try:
        raw = await medgemma_complete(system=ANALYSER_SYSTEM, user=user, temperature=0.2)
        analysis_dict = parse_json_block(raw)
        analysis = Analysis.model_validate(analysis_dict)
    except Exception as err:
        hpi: HPI | None = state.get("hpi")
        red_flags = hpi.redFlagsIdentified if hpi else []

        analysis = Analysis(
            understanding="The automated assessment could not be completed.",
            primaryAssessment={
                "condition": "No diagnostic assessment available",
                "probability": 0.0,
                "reasoning": "The diagnostic model could not be reached, so no assessment was produced. This is a system failure, not a finding of low risk.",
                "differentiatedFrom": []
            },
            considerations=[],
            redFlags=red_flags,
            suggestedUrgency="A_AND_E" if len(red_flags) > 0 else "NHS_111",
            reasoning=f"Analyser failure: {str(err)}. Defaulting to human clinical review.",
            confidence="LOW"
        )

    return {"analysis": analysis}
