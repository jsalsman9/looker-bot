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

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    sample_df = df.sample(n=min(5, len(df)), random_state=42)
    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items() if col in df.columns])

    planning_prompt = f"""
You are a data analyst assistant. Given a user question, a data dictionary, and sample data, generate Python code to help answer the question.

- Use only the columns described in the data dictionary.
- You may create multiple named variables if needed (e.g., campaign_summary, top_conversions).
- Do not print anything. Just define the variables.
- Output only executable Python code.

User question:
"{question}"

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_string(index=False)}
"""

    try:
        plan_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": planning_prompt}
            ],
            temperature=0.2
        )
        raw_code = plan_response.choices[0].message.content.strip()

        if raw_code.startswith("```"):
            raw_code = raw_code.strip("` ")
            if raw_code.startswith("python"):
                raw_code = raw_code[len("python"):].strip()

        local_vars = {"df": df.copy()}
        exec(raw_code, {}, local_vars)

        # Gather small result objects for LLM summary
        objects_for_summary = {}
        for name, val in local_vars.items():
            if name.startswith("_") or name == "df":
                continue
            if isinstance(val, (int, float, str)):
                objects_for_summary[name] = val
            elif isinstance(val, pd.DataFrame) and val.shape[0] <= 5:
                objects_for_summary[name] = val.to_markdown(index=False)

        if not objects_for_summary:
            return "✅ Code executed, but no readable result objects were produced."

        summary_prompt = f"""
You are a helpful data analyst. The user originally asked:
"{question}"

Here are the Python result objects that were created:
"""
        for k, v in objects_for_summary.items():
            summary_prompt += f"\n{k} =\n{v}\n"

        summary_prompt += "\nWrite a concise and helpful response based on this information."

        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": summary_prompt}
            ],
            temperature=0.3
        )

        return f"**Answer:** {summary_response.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ Error: {e}"
