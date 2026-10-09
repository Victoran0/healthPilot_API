from typing import Callable, List, Optional, Dict, Any
from pydantic import BaseModel
from validator import URGENCY_LADDER, Urgency, HPI, Analysis


class SafetyRule(BaseModel):
    id: str
    floor: Urgency
    reason: str
    matches: Callable[[str], bool]


def kw(*terms: str) -> Callable[[str], bool]:
    return lambda text: any(t in text for t in terms)


SAFETY_RULES: List[SafetyRule] = [
    SafetyRule(
        id="UNRESPONSIVE",
        floor="EMERGENCY_999",
        reason="Someone is unresponsive or cannot be woken",
        matches=kw(
            "unresponsive", "won't wake", "wont wake", "can't wake", "cant wake",
            "not breathing", "unconscious", "collapsed and", "passed out and won't"
        )
    ),
    SafetyRule(
        id="STROKE_FAST",
        floor="EMERGENCY_999",
        reason="Possible stroke in progress (FAST positive)",
        matches=lambda t: kw("face droop", "facial droop", "face has dropped")(t) or
        (kw("sudden")(t) and kw("slurred speech", "can't speak",
         "cannot speak", "arm weakness", "one side", "numbness on one")(t))
    ),
    SafetyRule(
        id="ANAPHYLAXIS",
        floor="EMERGENCY_999",
        reason="Possible anaphylaxis",
        matches=lambda t: kw("anaphyla", "throat closing",
                             "throat is closing", "tongue swelling", "tongue is swelling")(t)
    ),
    SafetyRule(
        id="SEVERE_BLEEDING",
        floor="EMERGENCY_999",
        reason="Severe uncontrolled bleeding",
        matches=kw(
            "won't stop bleeding", "wont stop bleeding", "uncontrolled bleeding",
            "bleeding heavily", "vomiting blood", "coughing up blood"
        )
    ),
    SafetyRule(
        id="CANNOT_BREATHE",
        floor="EMERGENCY_999",
        reason="Severe respiratory distress in progress",
        matches=kw(
            "can't breathe", "cant breathe", "cannot breathe", "struggling to breathe",
            "gasping", "blue lips", "turning blue"
        )
    ),
    SafetyRule(
        id="SELF_HARM_INTENT",
        floor="EMERGENCY_999",
        reason="Immediate risk of self-harm",
        matches=kw("kill myself", "end my life",
                   "going to hurt myself", "suicidal")
    )
]


def rank(u: Urgency) -> int:
    return URGENCY_LADDER.index(u)


class SafetyVerdict(BaseModel):
    floor: Urgency
    firedRules: List[SafetyRule]


def evaluate_safety_floor(hpi: Optional[HPI], _analysis: Optional[Analysis], transcript: str) -> SafetyVerdict:
    hpi_str = hpi.model_dump_json() if hpi else ""
    haystack = f"{transcript} {hpi_str}".lower()

    fired = [r for r in SAFETY_RULES if r.matches(haystack)]

    floor: Urgency = "SELF_CARE"
    for r in fired:
        if rank(r.floor) > rank(floor):
            floor = r.floor

    return SafetyVerdict(floor=floor, firedRules=fired)


def insufficient_data_floor(intake_sufficient: bool, analysis_usable: bool) -> Dict[str, Any]:
    if intake_sufficient and analysis_usable:
        return {"floor": "SELF_CARE", "reason": None}
    if not analysis_usable:
        return {
            "floor": "NHS_111",
            "reason": "The diagnostic models could not complete an assessment, so this recommendation is based on incomplete analysis rather than on a clinical finding."
        }
    return {
        "floor": "NHS_111",
        "reason": "Not enough information was gathered to run a reliable assessment, so this recommendation is cautious by default rather than based on a diagnosis."
    }


def apply_floor(model_urgency: Urgency, floor: Urgency) -> Dict[str, Any]:
    if rank(model_urgency) >= rank(floor):
        return {"urgency": model_urgency, "overridden": False}
    return {"urgency": floor, "overridden": True}
