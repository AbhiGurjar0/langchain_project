from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def ask_agent(query: str):

    # CASE 1: Visualization request
    if is_visualization_query(query):

        prompt = chart_prompt.format(columns=list(df.columns), request=query)

        response = chart_llm.invoke(prompt)

        response = clean_code(response)

        if is_chart_code(response):

            try:

                plt.clf()

                # safe numeric handling for heatmap/correlation
                safe_df = df.select_dtypes(include=["number"])

                exec(
                    response,
                    {
                        "df": (
                            safe_df
                            if "heatmap" in query.lower() or "corr" in query.lower()
                            else df
                        ),
                        "plt": plt,
                        "pd": pd,
                        "sns": sns,
                    },
                )

                plt.savefig("chart.png")
                plt.close()

                return {"type": "chart", "chart_path": "chart.png"}

            except Exception as e:

                return {"type": "text", "content": f"Chart generation failed: {str(e)}"}

    # CASE 2: Normal question → send to LLM with normal prompt
    else:

        context = get_dataset_context(df)

        prompt = normal_prompt.format(context=context, query=query)

        response = text_llm.invoke(prompt)

        return {"type": "text", "content": response.strip()}
