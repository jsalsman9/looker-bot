import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from gsheet_helper import load_sheet_data
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


def get_analysis_plan(question: str):
    prompt = f"""
You are a data planning assistant.

Your job is to convert user analysis questions into a JSON plan describing:
- any filters to apply (like dates, categories),
- columns to group by,
- and what metric to aggregate (e.g., sum, mean).

Only respond with a JSON array of steps. Do NOT include any text before or after.

Supported operations:
- filter_column, operation, value
- group_by
- agg_column, agg_func

Example input: "What campaigns performed best this year?"
Example output:
[
  {{"filter_column": "date", "operation": "contains", "value": "2024"}},
  {{"group_by": "campaign"}},
  {{"agg_column": "conversions", "agg_func": "sum"}}
]

Now process this user question:
"{question}"
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    # Get GPT-generated analysis plan
    plan = get_analysis_plan(question)
    st.write("🧠 GPT plan (raw):", plan)

    try:
        instructions = json.loads(plan)
        if not instructions or not isinstance(instructions, list):
            return "❌ GPT returned an empty or invalid analysis plan."
    except json.JSONDecodeError:
        return "❌ Failed to parse GPT plan as JSON."

    # Apply instructions to the DataFrame
    try:
        working_df = df.copy()

        for step in instructions:
            if "filter_column" in step:
                col = step["filter_column"]
                op = step["operation"]
                val = step["value"]

                if op == "equals":
                    working_df = working_df[working_df[col] == val]
                elif op == "contains":
                    working_df = working_df[working_df[col].astype(str).str.contains(val)]
                elif op == "greater_than":
                    working_df = working_df[pd.to_numeric(working_df[col], errors='coerce') > float(val)]
                elif op == "less_than":
                    working_df = working_df[pd.to_numeric(working_df[col], errors='coerce') < float(val)]

            elif "group_by" in step:
                group_col = step["group_by"]
                working_df = working_df.groupby(group_col, dropna=False)

            elif "agg_column" in step:
                agg_col = step["agg_column"]
                agg_func = step["agg_func"]
                if isinstance(working_df, pd.core.groupby.generic.DataFrameGroupBy):
                    working_df = working_df[agg_col].agg(agg_func).reset_index()
                else:
                    working_df = working_df[[agg_col]].agg(agg_func).to_frame().T

        if working_df.empty:
            return "❌ No data matched the criteria."

        # Ask GPT to explain the resulting table
        explanation_prompt = f"""
You are a helpful data analyst. A user asked this question:

{question}

Here is a summary table computed from a Google Sheet:

{working_df.head(20).to_string(index=False)}

Please provide a clear explanation of what this data shows.
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Failed to execute plan: {e}"
