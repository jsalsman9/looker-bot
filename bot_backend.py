import os
import json
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data

load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

from data_dictionary import data_dictionary, kpi_guide  # optional

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    sample_df = df.sample(n=min(5, len(df)), random_state=42)

    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items() if col in df.columns])

    planning_prompt = f"""
You are a helpful data analyst.
You are provided with:
- A sample of a dataset (`df`)
- A data dictionary describing each column
- A user question

You must return valid **Python code** to answer the user's question using ONLY the columns in the data.
- Use the variable `df` (a pandas DataFrame).
- You may define intermediate variables.
- You **must** end with:
  ```python
  results = {
      "metric1": value1,
      "metric2": value2
  }
  ```
- If a result is a DataFrame, assign it as a value (e.g., `"top_campaigns": top_df`).

🛑 Do NOT include any explanation or text — only Python code.

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_markdown(index=False)}

{kpi_guide}

User question: {question}
"""

    try:
        code_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": planning_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        code = code_response.choices[0].message.content.strip()
        if code.startswith("```"):
            code = code.strip("` ")
            if code.startswith("python"):
                code = code[len("python"):].strip()
    except Exception as e:
        return f"❌ Error during code generation: {e}"

    # Execute generated code
    local_vars = {}
    try:
        exec(code, {"df": df, "pd": pd}, local_vars)
        results = local_vars.get("results")
        if results is None:
            return "❌ Code did not assign `results` dictionary."
    except Exception as e:
        return f"❌ Error executing generated code: {e}\n\nCode was:\n```python\n{code}\n```"

    # Second LLM pass to summarize
    summary_prompt = f"""
You are a helpful data analyst.
The user originally asked:
"{question}"

Here are the results of the analysis:
{json.dumps({k: str(v) for k, v in results.items()}, indent=2)}

Please summarize the answer clearly and concisely for the user.
"""
    try:
        final_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": summary_prompt}
            ],
            temperature=0.3
        )
        return f"**Answer:** {final_response.choices[0].message.content.strip()}"
    except Exception as e:
        return f"✅ Executed code but failed to summarize: {e}\n\n{json.dumps(results, indent=2)}"
