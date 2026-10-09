from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Urgency = Literal[
    "SELF_CARE",
    "PHARMACIST",
    "GP_ROUTINE",
    "GP_URGENT",
    "NHS_111",
    "A_AND_E",
    "EMERGENCY_999"
]

URGENCY_LADDER = [
    "SELF_CARE",
    "PHARMACIST",
    "GP_ROUTINE",
    "GP_URGENT",
    "NHS_111",
    "A_AND_E",
    "EMERGENCY_999"
]


class Symptom(BaseModel):
    name: str
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[int] = Field(None, ge=0, le=10)
    character: Optional[str] = None
    radiation: Optional[str] = None
    aggravating: Optional[str] = None
    relieving: Optional[str] = None


class PatientProfile(BaseModel):
    ageYears: Optional[float] = None
    sex: Optional[Literal["male", "female"]] = None
    bmi: Optional[float] = None
    systolicBP: Optional[float] = None
    hba1c: Optional[float] = None
    totalCholesterol: Optional[float] = None
    smoker: Optional[bool] = None
    race: Optional[Literal["white", "black"]] = None
    ethnicity: Optional[Literal["french", "italian",
                                "chinese", "african", "german"]] = None


class HPI(BaseModel):
    chiefComplaint: str
    historyOfPresentIllness: str
    symptoms: List[Symptom]
    relevantNegatives: List[str]
    pastMedicalHistory: List[str]
    medications: List[str]
    allergies: List[str]
    familyHistory: List[str]
    socialHistory: List[str]
    availableImaging: List[str]
    availableLabs: List[str]
    redFlagsIdentified: List[str]
    informationStillNeeded: List[str]
    intakeConfidence: Literal["HIGH", "MODERATE", "LOW"]
    candidateConditions: List[str]
    patientProfile: PatientProfile
    cardioRespRelevant: bool


class Inquiry(BaseModel):
    question: Optional[str] = None
    rationale: str = ""
    requestedArtifacts: List[Literal["CHEST_XRAY", "ECG",
                                     "BLOODS", "ECHO", "NONE"]] = Field(default_factory=list)
    intakeComplete: bool = False


class ImagingLabel(BaseModel):
    pathology: str
    probability: float


class ImagingResult(BaseModel):
    model: str
    labels: List[ImagingLabel]
    topFindings: List[str]
    note: str


class RiskScore(BaseModel):
    outcome: str
    probability: float


class EhrResult(BaseModel):
    model: str
    riskScores: List[RiskScore]
    medicationFlags: List[str]
    note: str


class RagPassage(BaseModel):
    title: str
    snippet: str
    score: float
    source: str


class RagResult(BaseModel):
    query: str
    passages: List[RagPassage]


class PrimaryAssessment(BaseModel):
    condition: str = "Undetermined"
    probability: float = Field(default=0.0, ge=0, le=1)
    reasoning: str = ""
    differentiatedFrom: List[str] = Field(default_factory=list)


class Consideration(BaseModel):
    condition: str
    likelihood: Literal["LIKELY", "POSSIBLE", "UNLIKELY", "CANNOT_EXCLUDE"]
    supportingEvidence: List[str] = Field(default_factory=list)
    contradictingEvidence: List[str] = Field(default_factory=list)


class Analysis(BaseModel):
    understanding: str
    primaryAssessment: PrimaryAssessment
    considerations: List[Consideration]
    redFlags: List[str]
    suggestedUrgency: Urgency
    reasoning: str
    confidence: Literal["HIGH", "MODERATE", "LOW"]


class Diagnosis(BaseModel):
    condition: str
    probability: float = Field(ge=0, le=1)
    confidence: Literal["HIGH", "MODERATE", "LOW"]
    differentials: List[str]


class TriageDecision(BaseModel):
    urgency: Urgency
    headline: str
    actions: List[str]
    safetyNetting: List[str]
    overriddenByRules: bool
    overrideReason: Optional[str] = None
    patientMessage: str
    diagnosis: Optional[Diagnosis] = None
