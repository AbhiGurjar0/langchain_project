from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import matplotlib
from dotenv import load_dotenv
load_dotenv()

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
import time
from langchain_groq import ChatGroq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
)
# Load dataset once
DATA_PATH = "titanic.csv"
df = pd.read_csv(DATA_PATH)

unified_prompt = PromptTemplate.from_template(
    """
You are an intelligent Titanic dataset assistant and data visualization expert.

You have access to a pandas DataFrame called df with these columns:
{columns}

Dataset context:
{context}

User query:
{query}

Your task:

If the user is asking for visualization (chart, graph, histogram, plot, heatmap, scatter, distribution, etc):

→ Generate ONLY valid matplotlib or seaborn Python code  
→ Use ONLY the existing dataframe df  
→ Do NOT explain  
→ Do NOT print  
→ Do NOT use markdown  
→ Output ONLY executable Python code  

If the user is asking a normal question:

→ Generate ONLY a clear natural language answer  
→ Do NOT generate code  
→ Do NOT explain the process  
→ Answer directly  

IMPORTANT:
- Output must be EITHER code OR text
- NEVER output both
- NEVER explain what you are doing
"""
)
# structured_classifier = llm.with_structured_output(QueryType)


# Clean code
def clean_code(code: str):
    code = code.replace("```python", "")
    code = code.replace("```", "")
    return code.strip()


# Detect chart
def is_chart_code(response: str):
    return "plt." in response or "sns." in response or ".plot(" in response


def get_dataset_context(df):

    context = ""

    # basic info
    context += f"Total passengers: {len(df)}\n"

    # survival stats
    context += f"Survived count: {df['Survived'].sum()}\n"

    # gender stats
    if "Sex" in df.columns:
        gender_counts = df["Sex"].value_counts().to_dict()
        context += f"Gender counts: {gender_counts}\n"

    # age stats
    if "Age" in df.columns:
        context += f"Average age: {df['Age'].mean():.2f}\n"

    return context


def ask_agent(query: str):

    context = get_dataset_context(df)

    prompt = unified_prompt.format(
        columns=list(df.columns), context=context, query=query
    )

    response = llm.invoke(prompt).content

    response = clean_code(response)

    # If response contains matplotlib code → chart
    if is_chart_code(response):

        try:

            plt.clf()

            exec(response, {"df": df, "plt": plt, "pd": pd, "sns": sns})

            filename = f"chart_{int(time.time())}.png"

            plt.savefig(filename, bbox_inches="tight")

            plt.close()

            return {"type": "chart", "chart_path": filename}

        except Exception as e:

            return {"type": "text", "content": f"Chart generation failed: {str(e)}"}

    # Otherwise → text answer
    else:

        return {"type": "text", "content": response}
