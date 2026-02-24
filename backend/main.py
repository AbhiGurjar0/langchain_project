from fastapi import FastAPI
from agent import ask_agent
from pydantic import BaseModel
from tool import get_dataframe

app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/hi")
def home():
    return {"Greetings": "Hello Abhi Good morning"}


@app.post("/chat")
def chat(query: Query):
    response = ask_agent( query.question)
    return {"answer": response}
