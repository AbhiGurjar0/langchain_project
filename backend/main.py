from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
from agent import ask_agent
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Titanic agent running"}


@app.post("/chat")
def chat(query: Query):

    result = ask_agent(query.question)

    if result["type"] == "chart":
        chart_path = result["chart_path"]

        return {"type": "chart", "chart_url": f"{BACKEND_URL}/chart/{chart_path}"}

    else:

        return {"type": "text", "answer": result["content"]}


@app.get("/chart/{filename}")
def get_chart(filename: str):

    return FileResponse(filename)
