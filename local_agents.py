# Not real agents. Just testing pipeline with hard code #
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# ---- Request/response models matching your call_candidate payload ----

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    content: str


# ---- Agent 1: telemetry / performance specialist ----

@app.post("/agent/telemetry")
def telemetry_agent(request: ChatRequest) -> ChatResponse:
    user_msg = request.messages[-1].content if request.messages else ""
    reply = (
        "[TelemetryAgent] I specialize in telemetry monitoring and performance optimization. "
        f"You asked: {user_msg}"
    )
    return ChatResponse(content=reply)


# ---- Agent 2: math tutor ----

@app.post("/agent/math")
def math_agent(request: ChatRequest) -> ChatResponse:
    user_msg = request.messages[-1].content if request.messages else ""
    reply = (
        "[MathAgent] I specialize in math tutoring and problem solving. "
        f"You asked: {user_msg}"
    )
    return ChatResponse(content=reply)