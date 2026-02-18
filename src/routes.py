from fastapi import FastAPI
from pydantic import BaseModel

from src.agent_builder import build_ktc_react_agent

app = FastAPI(title="KTC Supply Chain API")
agent_executor = build_ktc_react_agent()


class Query(BaseModel):
    input: str


@app.post("/chat")
async def chat_endpoint(query: Query):
    result = await agent_executor.ainvoke({"input": query.input, "chat_history": []})
    return {"answer": result.get("output", "")}
