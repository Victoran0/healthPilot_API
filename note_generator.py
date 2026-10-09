import re
from llm import note_llm
from validator import HPI


SYSTEM_PROMPT = (
    "You are a Clinical Informatics Specialist. Your task is to synthesize structured data "
    "into a high-density SOAP note for downstream NLP extraction. "
    "1. TERMINOLOGY: Use standard clinical shorthand (e.g., PMH, NKDA, WNL, HTN, DM2). "
    "2. HEALTHY PATIENTS: If a patient has no documented chronic conditions and labs are "
    "within normal limits, explicitly state 'No acute clinical concerns' in the Assessment. "
    "3. SPARSE DATA: If a section is empty, do not say 'None'; use 'Unremarkable', 'Non-contributory', "
    "or 'Routine health maintenance' to provide clinical context. "
    "4. SYNTHESIS: Critically evaluate laboratory/vital signs. If labs are abnormal but not listed "
    "in the history, include the clinical impression in the 'Assessment' section. "
    "5. FORMAT: Output plain text only. No markdown, no bolding, no headers like '###'. "
    "Use a single line break between SOAP sections."
)


def build_patient_prompt(hpi: HPI) -> str:
    p = hpi.patientProfile

    demo = f"{round(p.ageYears)}yo {p.sex}{f', {p.race}' if p.race else ''}" if p.ageYears is not None and p.sex is not None else "adult, demographics not stated"
    pmh = "; ".join(
        hpi.pastMedicalHistory) if hpi.pastMedicalHistory else "Non-contributory"
    meds = "; ".join(
        hpi.medications) if hpi.medications else "No active medications"
    allergies = "; ".join(hpi.allergies) if hpi.allergies else "NKDA"

    lab_lines = []
    if p.bmi is not None:
        lab_lines.append(f"Body Mass Index: {p.bmi} kg/m2")
    if p.systolicBP is not None:
        lab_lines.append(f"Systolic Blood Pressure: {p.systolicBP} mmHg")
    if p.hba1c is not None:
        lab_lines.append(f"Hemoglobin A1c: {p.hba1c} %")
    if p.totalCholesterol is not None:
        lab_lines.append(f"Total Cholesterol: {p.totalCholesterol} mg/dL")
    labs = "\n".join(
        f"  {l}" for l in lab_lines) if lab_lines else "Vitals and labs unremarkable"

    symptom_parts = []
    for s in hpi.symptoms:
        parts = [s.name]
        if s.onset:
            parts.append(f"onset {s.onset}")
        if s.duration:
            parts.append(f"for {s.duration}")
        if s.severity is not None:
            parts.append(f"severity {s.severity}/10")
        symptom_parts.append(", ".join(parts))
    symptom_summary = "; ".join(symptom_parts)

    is_healthy = len(hpi.pastMedicalHistory) == 0 and len(
        hpi.medications) == 0 and len(lab_lines) == 0

    assessment_instruction = 'If no acute concern is evident, state "No acute clinical concerns".' if is_healthy else "Synthesize findings. If labs (like BP or BMI) are abnormal, list the clinical impression here."

    return f"""Generate a professional SOAP note for the following patient data:

PATIENT DATA:
- Demographics: {demo}
- Presenting complaint: {hpi.chiefComplaint}{f' ({symptom_summary})' if symptom_summary else ''}
- PMH: {pmh}
- Meds: {meds}
- Allergies: {allergies}
- Labs/Vitals:
{labs}
- Procedures: None

INSTRUCTIONS:
- S: Summarize demographics, the presenting complaint, and history.
- O: List objective findings.
- A: Assessment. {assessment_instruction}
- P: Note 'Continue current management' or specific follow-ups.
- Keep the total length under 250 words for optimal BERT encoding."""


def strip_reasoning(text: str) -> str:
    out = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    if re.search(r"<think>", out, re.IGNORECASE):
        match = re.search(r"S:\s", out)
        if match:
            out = out[match.start():]
    return out


def template_fallback(hpi: HPI) -> str:
    p = hpi.patientProfile
    demo = f"{round(p.ageYears)}yo {p.sex}" if p.ageYears is not None and p.sex is not None else "adult"
    pmh = "; ".join(
        hpi.pastMedicalHistory) if hpi.pastMedicalHistory else "Non-contributory"
    meds = "; ".join(
        hpi.medications) if hpi.medications else "No active medications"
    allergies = "; ".join(hpi.allergies) if hpi.allergies else "NKDA"

    return "\n".join([
        f"S: {demo} presents with {hpi.chiefComplaint}. {hpi.historyOfPresentIllness} PMH: {pmh}. Meds: {meds}. {allergies}.",
        "O: Vitals and labs unremarkable.",
        f"A: {hpi.chiefComplaint}; {', '.join(hpi.candidateConditions) or 'assessment in progress'}.",
        "P: Routine health maintenance and follow-up."
    ])


async def generate_soap_note(hpi: HPI) -> str:
    try:
        res = await note_llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_patient_prompt(hpi)}
        ])
        print(
            f"[HealthPilot] SOAP note generation succeeded, length {len(res.content)}")
        raw = str(res.content)
        note = strip_reasoning(raw).strip()
        if len(note) > 20:
            return note
        raise ValueError("Empty note from LLM")
    except Exception as err:
        print(
            f"[HealthPilot] SOAP note generation failed, using template fallback: {err}")
        return template_fallback(hpi)
