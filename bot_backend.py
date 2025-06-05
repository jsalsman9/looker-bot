import openai
import os
from openai import OpenAI
import pandas as pd
from gsheet_helper import load_sheet_data
from dotenv import load_dotenv
import streamlit as st
import json

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "Could not load data from the sheet."

    # 1. Let GPT plan the transformation
    planning_prompt = f"""
    You are a data analysis planner. Based on the user's question below, return a JSON with:
    - filters: any filters needed (e.g., date column must be in 2025)
    - group_by: column(s) to group by
    - metrics: list of metric(s) to compute (e.g., count, mean, sum of column X)
    - sort_by: column to sort results by
    - sort_order: asc or desc
    Only return valid JSON.

    User question: "{question}"
    Available columns: {list(df.columns)}
    """

    plan_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": planning_prompt}],
        temperature=0.3
    )

    try:
        plan = json.loads(plan_response.choices[0].message.content)
    except Exception as e:
        return f"Failed to parse GPT planning response: {e}"

    # 2. Execute the plan in pandas
    try:
        df_exec = df.copy()

        # Apply filters
        for f in plan.get("filters", []):
            # Very basic logic for now
            if "year" in f.lower():
                date_cols = [col for col in df.columns if "date" in col.lower()]
                if date_cols:
                    df_exec[date_cols[0]] = pd.to_datetime(df_exec[date_cols[0]], errors='coerce')
                    df_exec = df_exec[df_exec[date_cols[0]].dt.year == 2025]

        # Group and aggregate
        group_cols = plan.get("group_by", [])
        metrics = plan.get("metrics", [])

        agg_dict = {}
        for metric in metrics:
            if "sum(" in metric:
                col = metric.split("sum(")[-1].rstrip(")")
                agg_dict[col] = "sum"
            elif "mean(" in metric:
                col = metric.split("mean(")[-1].rstrip(")")
                agg_dict[col] = "mean"
            elif "count(" in metric:
                col = metric.split("count(")[-1].rstrip(")")
                agg_dict[col] = "count"

        df_summary = df_exec.groupby(group_cols).agg(agg_dict).reset_index()

        sort_col = plan.get("sort_by")
        if sort_col in df_summary.columns:
            df_summary = df_summary.sort_values(sort_col, ascending=plan.get("sort_order", "desc") == "asc")

        sample_text = df_summary.head(10).to_string(index=False)

    except Exception as e:
        return f"Failed to execute plan: {e}"

    # 3. Let GPT explain the results
    explanation_prompt = f"""
    You are a data analyst.

    The user asked: {question}

    Here is the result:

    {sample_text}

    Please explain the findings in clear English.
    """

    explanation_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": explanation_prompt}],
        temperature=0.4
    )

    return explanation_response.choices[0].message.content
