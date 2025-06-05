import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
from difflib import get_close_matches
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

def match_column(name, columns, cutoff=0.6):
    name = name.lower().strip()
    lowered = {col.lower(): col for col in columns}
    matches = get_close_matches(name, lowered.keys(), n=1, cutoff=cutoff)
    return lowered[matches[0]] if matches else None

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items()])
    system_prompt = f"""
You are a data analyst working with a dataset from Google Sheets.

The dataset has the following columns:
{dictionary_text}

Here are the first few rows of data:
{df.head(5).to_string(index=False)}

The user will ask questions about trends, outliers, comparisons, or summaries. Use the column descriptions to infer intent and return a structured JSON plan. Your plan can include:

- filter_column, filter_op, filter_value
- group_by
- agg_column, agg_func (like sum, mean, count)

Respond with a JSON list of steps.
"""

    try:
        plan_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )
        plan_json = plan_response.choices[0].message.content
        st.write("🧠 GPT plan (raw):", plan_json)
        instructions = json.loads(plan_json)
    except Exception as e:
        return f"❌ Failed to generate plan: {e}"

    working_df = df.copy()
    try:
        for step in instructions:
            if "filter_column" in step:
                col = match_column(step["filter_column"], df.columns)
                if col is None:
                    return f"❌ Column '{step['filter_column']}' not found."
                op = step.get("filter_op", "==")
                val = step["filter_value"]

                if op == "==":
                    working_df = working_df[working_df[col] == val]
                elif op == ">=":
                    working_df = working_df[working_df[col] >= val]
                elif op == "<":
                    working_df = working_df[working_df[col] < val]

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

    except Exception as e:
        return f"❌ Failed to execute plan: {e}"

    try:
        answer_prompt = f"""
Given the user question: "{question}"
and the resulting data:
{working_df.to_string(index=False)}

Write a helpful summary of the result.
"""
        answer = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize data results clearly."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.4
        )
        return answer.choices[0].message.content
    except Exception as e:
        return f"✅ Data computed, but failed to summarize: {e}"
