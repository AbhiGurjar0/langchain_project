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

        return {
            "type": "chart",
            "chart_url": "http://localhost:8000/chart"
        }

    else:

        return {
            "type": "text",
            "answer": result["content"]
        }


@app.get("/chart")
def get_chart():

    return FileResponse("chart.png")