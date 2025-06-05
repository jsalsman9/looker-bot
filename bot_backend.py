import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from difflib import get_close_matches
from gsheet_helper import load_sheet_data

load_dotenv()

# Authenticate OpenAI client
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def match_column(name, columns, cutoff=0.6):
    """Fuzzy match a column name from a list of columns."""
    name = name.lower().strip()
    lowered = {col.lower(): col for col in columns}
    matches = get_close_matches(name, lowered.keys(), n=1, cutoff=cutoff)
    return lowered[matches[0]] if matches else None

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
        return "Could not load data from the sheet."

    plan = get_analysis_plan(question)
    st.write("🧠 GPT plan (raw):", plan)

    try:
        instructions = json.loads(plan)
        if not instructions or not isinstance(instructions, list):
            return "❌ GPT returned an empty or invalid analysis plan."
    except json.JSONDecodeError:
        return "❌ Failed to parse GPT plan as JSON."

    working_df = df.copy()

    try:
        for step in instructions:
            if "filter_column" in step:
                col = match_column(step["filter_column"], df.columns)
                if col is None:
                    return f"❌ Column '{step['filter_column']}' not found in sheet."

                op = step["operation"]
                val = step["value"]

                if op == "equals":
                    working_df = working_df[working_df[col] == val]
                elif op == "contains":
                    working_df = working_df[working_df[col].astype(str).str.contains(str(val), case=False, na=False)]
                elif op == "greater_than":
                    working_df = working_df[pd.to_numeric(working_df[col], errors='coerce') > float(val)]
                elif op == "less_than":
                    working_df = working_df[pd.to_numeric(working_df[col], errors='coerce') < float(val)]

            elif "group_by" in step:
                group_col = match_column(step["group_by"], df.columns)
                if group_col is None:
                    return f"❌ Column '{step['group_by']}' not found."
                working_df = working_df.groupby(group_col)

            elif "agg_column" in step:
                agg_col = match_column(step["agg_column"], df.columns)
                if agg_col is None:
                    return f"❌ Column '{step['agg_column']}' not found."
                agg_func = step["agg_func"]

                if isinstance(working_df, pd.core.groupby.generic.DataFrameGroupBy):
                    result = working_df[agg_col].agg(agg_func).reset_index()
                    if agg_col in result.columns and agg_col in [step.get("group_by") for step in instructions]:
                        result = result.rename(columns={agg_col: f"{agg_func}_{agg_col}"})
                    working_df = result
                else:
                    result = working_df[[agg_col]].agg(agg_func).to_frame().T
                    result.columns = [f"{agg_func}_{agg_col}"]
                    working_df = result

        result_sample = working_df.head(10).to_string(index=False)

        final_prompt = f"""
You are a data analyst. Here is a table:

{result_sample}

Explain the result of the user's question: "{question}"
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Failed to execute plan: {e}"
