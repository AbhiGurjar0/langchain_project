from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
import time

# classifier model (fast model recommended)
classifier_llm = ChatOllama(model="llama3.1:8b", temperature=0)


class QueryType(TypedDict):
    type: Literal["0", "1"]  # 0 = chart, 1 = text


# Load dataset once
DATA_PATH = "titanic.csv"
df = pd.read_csv(DATA_PATH)

# Initialize LLM
chart_llm = OllamaLLM(model="deepseek-coder:6.7b", temperature=0)

text_llm = OllamaLLM(model="llama3.1:8b", temperature=0)

# Unified prompt

chart_prompt = PromptTemplate.from_template(
    """
You are a Python data visualization expert.

Data columns:
{columns}

User request:
{request}

Generate ONLY valid matplotlib Python code.

Rules:
- Use dataframe name: df
- Do NOT explain
- Do NOT print
- Only code
"""
)

normal_prompt = PromptTemplate.from_template(
    """
You are a Titanic dataset assistant.

Dataset context:
{context}

User question:
{query}

Rules:
- Answer using the dataset context
- Answer in clear natural language
- Do NOT generate code
- Give direct answer

Answer:
"""
)
classifier_prompt = """
You are a query classifier.

Classify the user query into one of two categories:

0 → Visualization request (chart, plot, graph, heatmap, visualize)
1 → Normal question (asking about data, statistics, percentage, count, etc.)

Return ONLY:

0 or 1

User query:
{query}
"""

structured_classifier = classifier_llm.with_structured_output(QueryType)


# Clean code
def clean_code(code: str):
    code = code.replace("```python", "")
    code = code.replace("```", "")
    return code.strip()


# Detect chart
def is_chart_code(response: str):
    return "plt." in response or "sns." in response or ".plot(" in response


# Detect if query is visualization request
def is_visualization_query(query: str):

    keywords = [
        "chart",
        "plot",
        "graph",
        "visualize",
        "visualization",
        "heatmap",
        "bar",
        "hist",
        "distribution",
        "scatter",
        "line",
    ]

    query = query.lower()

    return any(keyword in query for keyword in keywords)


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


def classify_query(query: str):

    result = structured_classifier.invoke(classifier_prompt.format(query=query))

    return result["type"]


def ask_agent(query: str):

    query_type = classify_query(query)

    # Case 0 → chart
    if query_type == "0":

        prompt = chart_prompt.format(columns=list(df.columns), request=query)

        code = chart_llm.invoke(prompt)

        code = clean_code(code)

        plt.clf()

        exec(code, {"df": df, "plt": plt, "pd": pd, "sns": sns})

        filename = f"chart_{int(time.time())}.png"

        plt.savefig(filename)
        plt.close()

        return {"type": "chart", "chart_path": filename}

    # Case 1 → normal answer
    else:

        context = get_dataset_context(df)

        prompt = normal_prompt.format(context=context, query=query)

        answer = text_llm.invoke(prompt)

        return {"type": "text", "content": answer}
