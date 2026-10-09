import re
from typing import Annotated, TypedDict, List, Optional, Dict, Any, Literal
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from validator import HPI, ImagingResult, EhrResult, RagResult, Analysis, TriageDecision, Inquiry
from analyzer_nodes import rag_node, chest_vision_node, ehr_node
from analyzer import analyser_node
from azure_agents import recipient_agent, inquirer_agent, triage_agent
from safety import evaluate_safety_floor, insufficient_data_floor, apply_floor, rank
from pathways import PATHWAYS


MAX_ROUNDS = 8


def append_list(left: list, right: list) -> list:
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class HealthPilotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    hpi: Optional[HPI]
    askedQuestions: Annotated[List[str], append_list]
    round: int
    xrayImageUrl: Optional[str]
    ehrRecord: Optional[Dict[str, Any]]
    imaging: Optional[ImagingResult]
    ehr: Optional[EhrResult]
    rag: Optional[RagResult]
    analysis: Optional[Analysis]
    triage: Optional[TriageDecision]


XRAY_UPLOAD_PROMPT = "Kindly click the attach button below to upload a picture of your X-ray."
XRAY_AVAILABILITY_QUESTION = "Have you had a chest X-ray, and can you access the image?"


def asked_about_xray(state: HealthPilotState) -> bool:
    return any(re.search(r"x-?ray", q, re.IGNORECASE) for q in state.get("askedQuestions", []))


def upload_already_requested(state: HealthPilotState) -> bool:
    return any("attach button" in q for q in state.get("askedQuestions", []))


def patient_has_xray(state: HealthPilotState) -> bool:
    hpi = state.get("hpi")
    if not hpi:
        return False
    return any(re.search(r"x-?ray", i, re.IGNORECASE) for i in hpi.availableImaging)


def needs_xray_availability_question(state: HealthPilotState) -> bool:
    hpi = state.get("hpi")
    return bool(hpi and hpi.cardioRespRelevant) and not asked_about_xray(state) and not state.get("xrayImageUrl")


def needs_xray_upload(state: HealthPilotState) -> bool:
    hpi = state.get("hpi")
    return bool(hpi and hpi.cardioRespRelevant) and patient_has_xray(state) and not state.get("xrayImageUrl") and not upload_already_requested(state)


async def recipient_node(state: HealthPilotState) -> dict:
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    patient_message = str(last.content) if last else ""

    xray_image_url = last.additional_kwargs.get(
        "xrayImageUrl") if last else None
    xray_image_url = xray_image_url or state.get("xrayImageUrl")

    ehr_record = last.additional_kwargs.get("ehrRecord") if last else None
    ehr_record = ehr_record or state.get("ehrRecord")

    asked_questions = state.get("askedQuestions", [])
    previous_question = asked_questions[-1] if asked_questions else "(none — patient opened the conversation)"

    hpi_str = state["hpi"].model_dump_json(
        indent=2) if state.get("hpi") else "(none — round 1)"

    prompt = f"""Previous HPI (H_{{t-1}}):
        {hpi_str}

        Question asked last round (q_{{t-1}}):
        {previous_question}

        Patient's latest message (D_t):
        {patient_message}"""

    # Native Microsoft Agent Framework execution with strict Pydantic enforcement
    response = await recipient_agent.run(
        prompt,
        options={"response_format": HPI}
    )

    # response.value is ALREADY a typed, validated HPI object!
    hpi: HPI = response.value

    if xray_image_url and not any(re.search(r"x-?ray", i, re.IGNORECASE) for i in hpi.availableImaging):
        hpi.availableImaging.append("Chest X-ray (uploaded)")

    return {
        "hpi": hpi,
        "round": state.get("round", 0) + 1,
        "xrayImageUrl": xray_image_url,
        "ehrRecord": ehr_record
    }


async def inquirer_node(state: HealthPilotState) -> dict:
    if needs_xray_upload(state):
        return {
            "messages": [AIMessage(content=XRAY_UPLOAD_PROMPT, additional_kwargs={"requests": ["CHEST_XRAY"], "phase": "intake"})],
            "askedQuestions": [XRAY_UPLOAD_PROMPT]
        }

    if needs_xray_availability_question(state):
        return {
            "messages": [AIMessage(content=XRAY_AVAILABILITY_QUESTION, additional_kwargs={"requests": [], "phase": "intake"})],
            "askedQuestions": [XRAY_AVAILABILITY_QUESTION]
        }

    hpi = state.get("hpi")
    asked_questions = state.get("askedQuestions", [])

    asked_questions_text = (
        "\n".join(f"{i+1}. {q}" for i, q in enumerate(asked_questions))
        if asked_questions
        else "(none)"
    )

    candidates_text = (
        ", ".join(hpi.candidateConditions)
        if hpi and hpi.candidateConditions
        else "(not yet established)"
    )
    hpi_json = hpi.model_dump_json(indent=2) if hpi else "(none)"
    provided_artifacts = "CHEST_XRAY" if state.get("xrayImageUrl") else "none"

    prompt = f"""Round {state.get('round', 0)} of {MAX_ROUNDS}
        Current HPI (H_t):
        {hpi_json}

        Candidate conditions to discriminate (d̂_t):
        {candidates_text}

        Questions already asked (Q_{{t-1}}) — DO NOT REPEAT ANY OF THESE:
        {asked_questions_text}

        Artefacts already provided: {provided_artifacts}"""

    # 1. Invoke agent
    # Enforce Inquiry schema
    response = await inquirer_agent.run(
        prompt,
        options={"response_format": Inquiry}
    )
    inquiry: Inquiry = response.value

    # 3. Build response
    requests = [] if state.get("xrayImageUrl") else [
        a for a in inquiry.requestedArtifacts if a != "NONE"]
    question_text = inquiry.question or ""

    return {
        "messages": [AIMessage(content=question_text, additional_kwargs={"requests": requests, "phase": "intake"})],
        "askedQuestions": [inquiry.question] if inquiry.question else []
    }


async def triage_node(state: HealthPilotState, config: RunnableConfig) -> dict:
    transcript = "\n".join([str(m.content) for m in state.get("messages", [])])

    safety_verdict = evaluate_safety_floor(
        state.get("hpi"), state.get("analysis"), transcript)
    emergency_floor = safety_verdict.floor
    fired_rules = safety_verdict.firedRules

    intake_sufficient = is_ready_for_diagnosis(state)
    analysis = state.get("analysis")
    analysis_usable = bool(
        analysis and analysis.primaryAssessment.probability > 0)

    data_floor_dict = insufficient_data_floor(
        intake_sufficient, analysis_usable)
    data_floor = data_floor_dict["floor"]
    data_reason = data_floor_dict["reason"]

    model_urgency = analysis.suggestedUrgency if analysis else "NHS_111"
    combined_floor = emergency_floor if rank(
        emergency_floor) >= rank(data_floor) else data_floor

    floor_result = apply_floor(model_urgency, combined_floor)
    urgency = floor_result["urgency"]
    overridden = floor_result["overridden"]

    override_reason = None
    if overridden:
        if fired_rules:
            override_reason = "; ".join(
                [f"{r.id}: {r.reason}" for r in fired_rules])
        else:
            override_reason = data_reason

    if overridden:
        print(
            f"[SAFETY] model={model_urgency} -> {urgency} :: {override_reason}")

    pathway = PATHWAYS[urgency]

    primary_assessment_str = "No diagnostic assessment available (the analyser did not complete)."
    if analysis and analysis.primaryAssessment:
        pa = analysis.primaryAssessment
        diff_str = ", ".join(
            pa.differentiatedFrom) if pa.differentiatedFrom else "none"
        primary_assessment_str = (
            f"{pa.condition} (estimated probability {pa.probability * 100:.0f}%, "
            f"overall confidence {analysis.confidence}). "
            f"Reasoning: {pa.reasoning} "
            f"Ranked below it: {diff_str}."
        )

    actions_text = "\n- ".join(pathway.actions) if pathway.actions else "none"
    safety_netting_text = "\n- ".join(
        pathway.safetyNetting) if pathway.safetyNetting else "none"

    if overridden:
        decided_by = "emergency safety rule" if fired_rules else "insufficient information to diagnose"
    else:
        decided_by = "the diagnostic analysis"

    data_gap = data_reason if (
        overridden and not fired_rules and data_reason) else "none"
    override_reason_text = override_reason or "n/a"
    analysis_json = analysis.model_dump_json(indent=2) if analysis else "null"
    hpi_json = state["hpi"].model_dump_json(
        indent=2) if state.get("hpi") else "null"

    human_msg = HumanMessage(content=f"""FINAL URGENCY (non-negotiable): {urgency}
        Decided by: {decided_by}
        Data gap (if any — explain this to the patient if present): {data_gap}
        Override reason (if any): {override_reason_text}
        DIAGNOSTIC ASSESSMENT (communicate this): {primary_assessment_str}
        Headline the patient sees: {pathway.headline}
        Actions rendered beneath your message (do NOT repeat or reference these):
        - {actions_text}
        Safety-netting rendered beneath (do NOT repeat):
        - {safety_netting_text}
        Full clinical analysis:
        {analysis_json}
        Patient's structured history:
        {hpi_json}""")

    # Invoke the Foundry Agent
    result = await triage_agent.ainvoke(
        {"messages": [human_msg]},
        config={
            **config,
            "tags": [*(config.get("tags") or []), "patient_facing"],
        },
    )
    print(f"[HealthPilot] triageNode raw output: {str(result)}")
    msgs = result["messages"]
    ai_msg = msgs[-1] if isinstance(msgs, list) else msgs

    diagnosis_dict = None
    if analysis and analysis.primaryAssessment:
        diagnosis_dict = {
            "condition": analysis.primaryAssessment.condition,
            "probability": analysis.primaryAssessment.probability,
            "confidence": analysis.confidence,
            "differentials": analysis.primaryAssessment.differentiatedFrom
        }

    triage = TriageDecision(
        urgency=urgency,
        headline=pathway.headline,
        actions=pathway.actions,
        safetyNetting=pathway.safetyNetting,
        overriddenByRules=overridden,
        overrideReason=override_reason,
        patientMessage=str(ai_msg.content),
        diagnosis=diagnosis_dict
    )

    return {
        "triage": triage,
        "messages": [AIMessage(content=str(ai_msg.content), additional_kwargs={"phase": "triage", "urgency": urgency})]
    }


def is_ready_for_diagnosis(state: HealthPilotState) -> bool:
    hpi = state.get("hpi")
    if not hpi:
        return False

    p = hpi.patientProfile
    has_identity = p.ageYears is not None and p.sex is not None
    has_complaint = len(hpi.symptoms) > 0 and any(
        s.onset or s.duration for s in hpi.symptoms)

    asked = {q.lower() for q in state.get("askedQuestions", [])}

    def asked_about(pattern: str) -> bool:
        return any(re.search(pattern, q) for q in asked)

    pmh_covered = len(hpi.pastMedicalHistory) > 0 or asked_about(
        r"medical condition|diagnos|past medical|pmh|history")
    meds_covered = len(hpi.medications) > 0 or asked_about(
        r"medication|medicine|taking|drug")
    allergy_covered = len(hpi.allergies) > 0 or any(re.search(
        r"allerg", n, re.IGNORECASE) for n in hpi.relevantNegatives) or asked_about(r"allerg")

    has_history_sweep = pmh_covered and meds_covered and allergy_covered
    identity_and_complaint = has_identity and has_complaint

    if needs_xray_availability_question(state) or needs_xray_upload(state):
        return False

    return identity_and_complaint and has_history_sweep


def route_phase(state: HealthPilotState) -> Literal["inquirerNode", "diagnose"]:
    if state.get("round", 0) >= MAX_ROUNDS:
        return "diagnose"
    if is_ready_for_diagnosis(state):
        return "diagnose"
    return "inquirerNode"


async def diagnose_node(state: HealthPilotState) -> dict:
    return {}

checkpointer = MemorySaver()

graph_builder = StateGraph(HealthPilotState)

graph_builder.add_node("recipientNode", recipient_node)
graph_builder.add_node("inquirerNode", inquirer_node)
graph_builder.add_node("diagnose", diagnose_node)
graph_builder.add_node("chestVision", chest_vision_node)
graph_builder.add_node("ehrNode", ehr_node)
graph_builder.add_node("ragNode", rag_node)
graph_builder.add_node("analyserNode", analyser_node)
graph_builder.add_node("triageNode", triage_node)

graph_builder.add_edge(START, "recipientNode")
graph_builder.add_conditional_edges("recipientNode", route_phase, {
    "inquirerNode": "inquirerNode",
    "diagnose": "diagnose"
})
graph_builder.add_edge("inquirerNode", END)

graph_builder.add_edge("diagnose", "chestVision")
graph_builder.add_edge("diagnose", "ehrNode")
graph_builder.add_edge("diagnose", "ragNode")

graph_builder.add_edge("chestVision", "analyserNode")
graph_builder.add_edge("ehrNode", "analyserNode")
graph_builder.add_edge("ragNode", "analyserNode")

graph_builder.add_edge("analyserNode", "triageNode")
graph_builder.add_edge("triageNode", END)

graph = graph_builder.compile(checkpointer=checkpointer)
