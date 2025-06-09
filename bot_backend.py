import os
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
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

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    sample_df = df.sample(n=min(3, len(df)), random_state=42)

    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items() if col in df.columns])

    planning_prompt = f"""
You are a Python data analyst. You will be given:
- A user question
- A data dictionary
- A preview of the dataset
- A guide to common marketing KPIs

Your task is to write Python code using ONLY the provided dataframe `df` to answer the user's question. Do not generate results yourself — only provide Python code.

⚠️ Guidelines:
- The dataframe is already named `df`
- ✅ Use pandas and numpy only — numpy is available as `np`
- ❌ Do NOT import anything
- ❌ Do NOT print() anything — return your result as a variable `result`
- ✅ Create and return ONE variable: `result`
- `result` can be a DataFrame, Series, number, string, etc.

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_string(index=False)}

{kpi_guide}

User question:
{question}
"""

    try:
        plan_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": planning_prompt}
            ],
            temperature=0.3
        )

        generated_code = plan_response.choices[0].message.content.strip()

        if generated_code.startswith("```"):
            generated_code = generated_code.strip("` ")
            if generated_code.startswith("python"):
                generated_code = generated_code[len("python"):].strip()

        local_vars = {"df": df.copy(), "np": np}
        exec(generated_code, {}, local_vars)

        result = local_vars.get("result")
        if result is None:
            return "❌ No result was returned by the code."

        explanation_prompt = f"""
You are a helpful data analyst. The user asked:
"{question}"

Here is the result of your analysis:
{str(result)[:2000]}

Write a clear, concise answer for the user based on this result.
"""

        explain_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": explanation_prompt}
            ],
            temperature=0.3
        )

        return f"**Answer:** {explain_response.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ Error: {e}"
