import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
import difflib
import json

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

data_dictionary = {
    "Campaign": "The marketing campaign name. This is lengthy and will contain codes and other alpha numerics",
    "Clicks": "Number of times the ad was clicked",
    "Impressions": "Number of times the ad was shown",
    "Media Cost": "Total ad spend in USD",
    "Date": "The date the ad was served (YYYY-MM-DD)",
    "Total Conversions": "Number of desired outcomes (e.g., signups, purchases)",
    "advertiser": "The client that this table belongs to. Will be the same value",
    "Activity ID": "Unique identifier for certain activities. Not all activities will have a set ID. Will not be important for analysis",
    "Placement": "Where this ad was placed. This is a very specific and long alpha numeric that contains where the ad was place, what type of ad was placed, and even the layout size of the ad",
    "Video Plays": "The amount of times a video was played for that ad",
    "video_completions": "The amount of times a user has completed a ad video in it's entirety"
}

kpi_guide = """
When asked which campaign is "performing the best", consider metrics like:
- Conversions (higher is better)
- Click-through rate (CTR = Clicks / Impressions)
- Cost per acquisition (CPA = Spend / Conversions, lower is better)
- Return on ad spend (ROAS = Revenue / Spend)

Choose a metric based on what is available in the data. If multiple apply, pick the most meaningful one and explain why.
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

    sample_df = df.sample(n=min(3, len(df)), random_state=42)

    essential_columns = [
        "Campaign", "Clicks", "Impressions", "Media Cost",
        "Date", "Total Conversions"
    ]
    dictionary_text = "\n".join([
        f"{col}: {data_dictionary[col]}"
        for col in essential_columns if col in data_dictionary
    ])

    system_prompt = f"""
You are a data planner. Given a user question and the dataset description, return a JSON list of steps to answer it.

Data dictionary:
{dictionary_text}

Example data:
{sample_df.to_string(index=False)}

{kpi_guide}

Return the plan in JSON only. Do not explain it. If unsure, guess reasonably.
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
        plan = json.loads(plan_response.choices[0].message.content)
        df_result = apply_plan(df, plan)
    except Exception as e:
        return f"❌ Failed to execute plan: {e}"

    return df_result.to_markdown(index=False) if not df_result.empty else "No results found."
