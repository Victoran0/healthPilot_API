import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage
import uvicorn

from agent import graph

app = FastAPI(title="HealthPilot API", version="1.0.0")


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    xray_image_url: Optional[str] = None
    ehr_record: Optional[Dict[str, Any]] = None


NODE_LABELS = {
    "recipientNode": "Understanding your symptoms…",
    "inquirerNode": "Working out what else to ask…",
    "chestVision": "Reading your chest X-ray…",
    "ehr": "Reviewing your medical history…",
    "rag": "Checking medical references…",
    "analyserNode": "Thinking through the possibilities…",
    "triageNode": "Deciding what you should do next…",
}


def sse(obj: dict) -> str:
    # jsonable_encoder handles pydantic models, datetimes, etc. so json.dumps
    # never blows up on a state value that isn't a plain dict.
    return f"data: {json.dumps(jsonable_encoder(obj))}\n\n"


async def event_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    msg = HumanMessage(
        content=req.message,
        additional_kwargs={
            "xrayImageUrl": req.xray_image_url,
            "ehrRecord": req.ehr_record,
        },
    )
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        # Only the newest message goes in. The checkpointer holds H_{t-1},
        # askedQuestions and round, same as your in-process JS version.
        async for event in graph.astream_events(
            {"messages": [msg]}, config, version="v2"
        ):
            kind = event["event"]
            name = event.get("name", "")
            tags = event.get("tags") or []
            data = event.get("data") or {}

            # A. Patient-facing streamed text. Allow-list on the tag, exactly
            # as route.ts does, so Recipient/Inquirer tokens don't leak.
            if kind == "on_chat_model_stream" and "patient_facing" in tags:
                chunk = data.get("chunk")
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    yield sse({"type": "text", "delta": content})

            # B. Structured output on node completion.
            elif kind == "on_chain_end":
                out = data.get("output") or {}
                get = out.get if isinstance(
                    out, dict) else (lambda k, d=None: d)

                if name == "inquirerNode":
                    m = get("messages") or []
                    question = getattr(m[0], "content", None) if m else None
                    requests = (
                        getattr(m[0], "additional_kwargs", {}).get("requests")
                        if m else None
                    ) or []
                    if isinstance(question, str) and question:
                        yield sse({"type": "text", "delta": question})
                    # Always emit, even empty. That's what clears the attach button.
                    yield sse({"type": "artefact-request", "requests": requests})

                elif name == "recipientNode":
                    hpi = get("hpi")
                    if hpi:
                        yield sse({"type": "hpi", "data": hpi})

                elif name in ("chestVision", "ehr", "rag"):
                    payload = get("imaging") or get("ehr") or get("rag")
                    if payload:
                        yield sse({"type": "evidence", "node": name, "data": payload})

                elif name == "analyserNode":
                    analysis = get("analysis")
                    if analysis:
                        yield sse({"type": "analysis", "data": analysis})

                elif name == "triageNode":
                    triage = get("triage")
                    if triage:
                        yield sse({"type": "triage", "data": triage})

            # C. Progress ticker.
            elif kind == "on_chain_start" and name in NODE_LABELS:
                yield sse({"type": "status", "node": name, "label": NODE_LABELS[name]})

    except Exception as e:
        # Fail loud, never reassuring. Mirror of your catch block.
        yield sse({
            "type": "text",
            "delta": (
                "Sorry, something went wrong on my side and I couldn't finish "
                "assessing your symptoms. Please don't take that as reassurance. "
                "If you're worried right now, contact NHS 111 (call 111 or go to "
                "111.nhs.uk), or call 999 if it feels like an emergency."
            ),
        })
        yield sse({"type": "error", "detail": str(e)})
    finally:
        yield sse({"type": "done"})


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        event_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx/proxy buffering the stream
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
