from typing import List, Literal
from pydantic import BaseModel
from validator import Urgency


class Pathway(BaseModel):
    urgency: Urgency
    headline: str
    actions: List[str]
    safetyNetting: List[str]
    severity: Literal["critical", "urgent", "moderate", "routine", "low"]


PATHWAYS: dict[Urgency, Pathway] = {
    "EMERGENCY_999": Pathway(
        urgency="EMERGENCY_999",
        headline="This is a medical emergency. Call 999 now.",
        actions=[
            "Call 999 immediately.",
            "Stay where you are and do not drive yourself.",
            "Unlock your front door if you can, so paramedics can reach you.",
            "If you have been prescribed a GTN spray or an adrenaline auto-injector, use it as directed."
        ],
        safetyNetting=[
            "If you become drowsy, confused, or struggle to stay awake, tell the 999 call handler.",
            "If someone is with you, ask them to stay with you until help arrives."
        ],
        severity="critical"
    ),
    "A_AND_E": Pathway(
        urgency="A_AND_E",
        headline="You need to be seen in A&E today.",
        actions=[
            "Go to your nearest A&E department now.",
            "Ask someone to take you, or call 111 if you cannot get there safely.",
            "Bring a list of your current medications."
        ],
        safetyNetting=[
            "If your symptoms get suddenly worse on the way, call 999.",
            "Do not wait to see if it settles on its own."
        ],
        severity="critical"
    ),
    "NHS_111": Pathway(
        urgency="NHS_111",
        headline="Contact NHS 111 today for urgent advice.",
        actions=[
            "Call 111, or go to 111.nhs.uk.",
            "They are open 24 hours a day and can arrange an urgent appointment if you need one.",
            "Have your symptoms and medications to hand."
        ],
        safetyNetting=[
            "Call 999 instead if you develop chest pain, severe breathlessness, confusion, or you cannot stay awake."
        ],
        severity="urgent"
    ),
    "GP_URGENT": Pathway(
        urgency="GP_URGENT",
        headline="Contact your GP practice today.",
        actions=[
            "Ring your GP practice as soon as it opens and ask for an urgent, same-day appointment.",
            "Tell reception exactly which symptoms you have — it affects how quickly you are seen.",
            "If the practice is closed, call NHS 111 instead."
        ],
        safetyNetting=[
            "If you cannot get an appointment today, call 111 rather than waiting.",
            "Call 999 if your symptoms become severe."
        ],
        severity="moderate"
    ),
    "GP_ROUTINE": Pathway(
        urgency="GP_ROUTINE",
        headline="Submit a GP request form.",
        actions=[
            "Fill in your GP practice's online request form (most UK practices call this an eConsult or online triage form).",
            "You can normally expect a response within a week — this is the right route for something that needs looking at but is not urgent.",
            "Note when your symptoms happen and what makes them better or worse, and include that in the form."
        ],
        safetyNetting=[
            "If your symptoms get worse while you wait, call your practice or NHS 111 rather than waiting for the form response.",
            "New or rapidly worsening symptoms mean you should be seen sooner."
        ],
        severity="routine"
    ),
    "PHARMACIST": Pathway(
        urgency="PHARMACIST",
        headline="A pharmacist can help with this.",
        actions=[
            "Visit any community pharmacy — no appointment needed.",
            "Under Pharmacy First, pharmacists can now treat several common conditions directly.",
            "Tell them what medications you already take."
        ],
        safetyNetting=[
            "See your GP if it has not improved within a week, or sooner if it worsens."
        ],
        severity="low"
    ),
    "SELF_CARE": Pathway(
        urgency="SELF_CARE",
        headline="This can usually be managed at home.",
        actions=[
            "Rest, keep hydrated, and monitor how you feel.",
            "Over-the-counter remedies from any pharmacy or supermarket can help with symptom relief — a pharmacist can point you to the right one.",
            "Keep track of whether things are improving day to day."
        ],
        safetyNetting=[
            "Contact your GP or NHS 111 if things get worse, do not improve as expected, or you become worried."
        ],
        severity="low"
    )
}
