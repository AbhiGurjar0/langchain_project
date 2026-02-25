from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

from agent import ask_agent


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

        return {
            "type": "chart",
            "chart_url": f"http://localhost:8000/chart/{chart_path}"
        }

    else:

        return {
            "type": "text",
            "answer": result["content"]
        }


@app.get("/chart/{filename}")
def get_chart(filename: str):

    return FileResponse(filename)