from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import CSVLoader
import os
from tool import get_dataframe

## embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://127.0.0.1:11434"
)

## llm
chat_model = ChatOllama(model="llama3", base_url="http://127.0.0.1:11434")

##prmopt

prompt = PromptTemplate(
    template="""
You are a professional Titanic dataset assistant.

STRICT INSTRUCTIONS:

1. Always answer in a clear, natural, complete sentence.
2. Include the subject of the question in your answer.
3. Do NOT show calculations, steps, or explanations unless the user explicitly asks for them using words like "explain", "process", "steps", or "how".
4. Do NOT include phrases like "The final answer is", "Based on the dataset", or similar unnecessary text.
5. Do NOT output raw values only. Always convert them into a natural sentence.
6. If the answer contains multiple values, present them clearly in one sentence.
7. If the answer is not present in the dataset, reply EXACTLY:
   This is not in my dataset

Dataset context:
{context}

User question:
{query}

Professional natural language answer:
""",
    input_variables=["context", "query"],
)

##parser
parser = StrOutputParser()


## CSV Loader
loader = CSVLoader(file_path="titanic.csv")
docs = loader.load()


## main llm calling function
def ask_agent(query):
    chain = prompt | chat_model | parser
    response = chain.invoke({"context": docs, "query": query})
    return {"response": response}
