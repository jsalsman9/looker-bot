import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
from difflib import get_close_matches
import json
import traceback

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

data_dictionary = {
    "Campaign": "The marketing campaign name...",
    "Clicks": "Number of times the ad was clicked",
    "Impressions": "Number of times the ad was shown",
    "Media Cost": "Total ad spend in USD",
    "Date": "The date the ad was served (YYYY-MM-DD)",
    "Total Conversions": "Number of desired outcomes (e.g., signups, purchases)",
    "advertiser": "The client that this table belongs to",
    "Activity ID": "Unique identifier for certain activities",
    "Placement": "Where this ad was placed",
    "Video Plays": "Times a video was played",
    "video_completions": "Times a video was fully viewed"
}

kpi_guide = """
When asked which campaign is "performing the best", consider metrics like:
- Conversions (higher is better)
- Click-through rate (CTR = Clicks / Impressions)
- Cost per acquisition (CPA = Spend / Conversions, lower is better)
- Return on ad spend (ROAS = Revenue / Spend)
"""

def fuzzy_match_column(requested_col, actual_cols):
    matches = get_close_matches(requested_col, actual_cols, n=1, cutoff=0.6)
    return matches[0] if matches else requested_col

def apply_plan(df, plan):
    if df.empty:
        raise ValueError("DataFrame is empty")

    for step in plan:
        try:
            if "filter" in step:
                col = fuzzy_match_column(step["filter"]["column"], df.columns)
                val = step["filter"]["value"]
                df = df[df[col] == val]

            elif "group_by" in step:
                col = fuzzy_match_column(step["group_by"], df.columns)
                df = df.groupby(col, as_index=False).first()

            elif "agg_column" in step:
                agg_col = fuzzy_match_column(step["agg_column"], df.columns)
                agg_func = step.get("agg_func", "sum")
                non_agg_cols = [col for col in df.columns if col != agg_col]
                group_col = non_agg_cols[0] if non_agg_cols else agg_col
                df = df.groupby(group_col, as_index=False).agg({agg_col: agg_func})

            elif "derive_column" in step:
                new_col = step["derive_column"]
                formula = step["formula"]
                df.eval(f"{new_col} = {formula}", inplace=True)

            elif "sort_by" in step:
                sort_col = fuzzy_match_column(step["sort_by"], df.columns)
                order = step.get("sort_order", "desc") == "desc"
                df = df.sort_values(by=sort_col, ascending=not order)

            elif "limit" in step:
                df = df.head(step["limit"])
        except Exception as inner_e:
            raise ValueError(f"Failed step {step}: {inner_e}")

    return df

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "Could not load data from the sheet."

    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items()])
    system_prompt = f"""
You are a data planner. Given a user question and the dataset description, return a JSON list of steps to answer it.

Data dictionary:
{dictionary_text}

Example data:
{df.head(5).to_string(index=False)}

{kpi_guide}

Return only the plan in JSON format. No explanation or extra text.
"""

    try:
        plan_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )
        raw_plan = plan_response.choices[0].message.content
        plan = json.loads(raw_plan)
    except Exception as plan_error:
        return f"❌ Failed to interpret plan: {plan_error}\n\nRaw GPT output:\n{raw_plan}"

    try:
        df_result = apply_plan(df, plan)
        return df_result.to_markdown(index=False) if not df_result.empty else "No results found."
    except Exception as e:
        return f"❌ Failed to execute plan: {e}\n\nPlan was:\n```json\n{json.dumps(plan, indent=2)}\n```"
