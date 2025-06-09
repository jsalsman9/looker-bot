import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
import json

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Example data dictionary – customize this as needed
data_dictionary = {
    "Campaign": "The marketing campaign name",
    "Clicks": "Number of times the ad was clicked",
    "Impressions": "Number of times the ad was shown",
    "Media Cost": "Total ad spend in USD",
    "Date": "The date the ad was served (YYYY-MM-DD)",
    "Total Conversions": "Number of desired outcomes (e.g., signups, purchases)"
}

kpi_guide = """
When asked which campaign is "performing the best", consider metrics like:
- Conversions (higher is better)
- Click-through rate (CTR = Clicks / Impressions)
- Cost per acquisition (CPA = Spend / Conversions, lower is better)
- Return on ad spend (ROAS = Revenue / Spend)
"""

def apply_plan(df, plan):
    for step in plan:
        if "filter" in step:
            col = step["filter"]["column"]
            val = step["filter"]["value"]
            df = df[df[col] == val]

        elif "group_by" in step:
            col = step["group_by"]
            df = df.groupby(col, as_index=False).first()

        elif "agg_column" in step:
            agg_col = step["agg_column"]
            agg_func = step.get("agg_func", "sum")
            group_col = df.columns[0] if df.columns[0] != agg_col else df.columns[1]
            df = df.groupby(group_col, as_index=False).agg({agg_col: agg_func})

        elif "derive_column" in step:
            new_col = step["derive_column"]
            formula = step["formula"]
            df.eval(f"{new_col} = {formula}", inplace=True)

        elif "sort_by" in step:
            sort_col = step["sort_by"]
            order = step.get("sort_order", "desc") == "desc"
            df = df.sort_values(by=sort_col, ascending=not order)

        elif "limit" in step:
            df = df.head(step["limit"])

        elif "code" in step:
            local_vars = {"df": df.copy()}
            try:
                exec(step["code"], {}, local_vars)
                if "df" in local_vars:
                    df = local_vars["df"]
                elif "result" in local_vars:
                    df = pd.DataFrame({"Result": local_vars["result"]})
            except Exception as e:
                raise ValueError(f"Failed to execute custom code step: {e}")

    return df

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    essential_columns = [
        "Campaign", "Clicks", "Impressions", "Media Cost",
        "Date", "Total Conversions"
    ]
    filtered_cols = [col for col in essential_columns if col in df.columns]
    sample_df = df[filtered_cols].sample(n=min(3, len(df)), random_state=42)

    dictionary_text = "\n".join([
        f"{col}: {data_dictionary[col]}" for col in filtered_cols if col in data_dictionary
    ])

    system_prompt = f"""
You are a data planner bot. Your job is to output a JSON plan that a Python backend will execute to answer the user's question.

You will be given:
- A data dictionary (describing each column)
- A preview of the dataset (sample rows)
- A guide to common KPIs

Your output must be a **valid JSON list of dictionaries** representing executable steps.
Each step should use one of these keys (only one per step):
- "filter": {{"column": "...", "value": "..."}}
- "group_by": "column_name"
- "agg_column": "column_name", "agg_func": "sum" | "mean" | "count" | "nunique"
- "derive_column": "new_column", "formula": "Clicks / Impressions"
- "sort_by": "column_name", "sort_order": "asc" | "desc"
- "limit": number
- "code": "Python code string that takes 'df' as input and updates or creates it"

🛑 Only return valid JSON (no explanation).
⚠️ If the user asks something like "What are the campaign names?", you   
  {{ "code": "df = pd.DataFrame({{'Campaign': df['Campaign'].unique()}})" }}

Data dictionary:
{dictionary_text}

Sample dataset:
{sample_df.to_string(index=False)}

{kpi_guide}
"""

    plan_response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )

    try:
        raw_plan = plan_response.choices[0].message.content.strip()
        if raw_plan.startswith("```"):
            raw_plan = raw_plan.strip("` ")
            if raw_plan.startswith("json"):
                raw_plan = raw_plan[4:].strip()

        plan = json.loads(raw_plan)
        df_result = apply_plan(df, plan)

        summary_prompt = f"""
You are a helpful data analyst. The user originally asked:

"{question}"

Here are the summarized results:

{df_result.to_markdown(index=False)}

Write a concise answer based on this result.
"""

        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "system", "content": summary_prompt}],
            temperature=0.3
        )

        return f"**Answer:** {summary_response.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ Error: {e}"
