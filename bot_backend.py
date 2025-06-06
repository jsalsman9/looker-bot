import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
import difflib
import json
import time

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
    "advertiser": "The client this table belongs to",
    "Activity ID": "Unique identifier, not important for analysis",
    "Placement": "Where the ad was placed",
    "Video Plays": "Number of video plays",
    "video_completions": "Video completions"
}

kpi_guide = """
When asked which campaign is "performing the best", consider:
- Conversions (higher = better)
- CTR = Clicks / Impressions
- CPA = Spend / Conversions (lower = better)
- ROAS = Revenue / Spend
Pick the best available metric.
"""

def fuzzy_match_column(requested_col, actual_cols):
    matches = difflib.get_close_matches(requested_col, actual_cols, n=1, cutoff=0.6)
    return matches[0] if matches else requested_col

def apply_plan(df, plan):
    for step in plan:
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
            group_col = df.columns[0] if df.columns[0] != agg_col else df.columns[1]
            df = df.groupby(group_col, as_index=False).agg({agg_col: agg_func})

        elif "derive_column" in step:
            new_col = step["derive_column"]
            formula = step["formula"]
            try:
                df.eval(f"{new_col} = {formula}", inplace=True)
            except Exception as e:
                raise ValueError(f"Failed to compute derived column {new_col}: {e}")

        elif "sort_by" in step:
            sort_col = fuzzy_match_column(step["sort_by"], df.columns)
            order = step.get("sort_order", "desc") == "desc"
            df = df.sort_values(by=sort_col, ascending=not order)

        elif "limit" in step:
            df = df.head(step["limit"])

    return df

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    # Use sample for planning
    sample_df = df.sample(min(len(df), 1000), random_state=42)

    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items()])
    system_prompt = f"""
You are a data planner. Given a user question and the dataset description, return a JSON list of steps to answer it.

Data dictionary:
{dictionary_text}

Example data:
{sample_df.head(5).to_string(index=False)}

{kpi_guide}

Return JSON only. No explanation.
"""

    try:
        plan_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            timeout=20  # add timeout
        )
        plan = json.loads(plan_response.choices[0].message.content)
    except Exception as e:
        return f"❌ GPT planning failed: {e}"

    try:
        start = time.time()
        df_result = apply_plan(df, plan)
        elapsed = time.time() - start
        if elapsed > 10:
            return "⚠️ Operation took too long and may need simplification."
    except Exception as e:
        return f"❌ Failed to execute plan: {e}"

    return df_result.to_markdown(index=False) if not df_result.empty else "No results found."

