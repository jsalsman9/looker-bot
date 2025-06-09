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

    sample_df = df.sample(n=min(5, len(df)), random_state=42)
    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items()])

    planning_prompt = f"""
You are a Python data analyst assistant. Your job is to write Python code to analyze the dataset and answer the user's question.

You will receive:
1. A data dictionary describing the columns
2. A preview of the dataset
3. The user's question

Your job is to write valid Python code that uses the variable `df` (a pandas DataFrame) to answer the question. Save the final result in a variable called `result`.
Do not write explanations or comments. Only output code. Use only the columns available.

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_string(index=False)}

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
        code = plan_response.choices[0].message.content.strip()
        if code.startswith("```"):
            code = code.strip("` \n")
            if code.startswith("python"):
                code = code[len("python"):].strip()

        if not code:
            return "❌ GPT returned empty code."

        print("🔧 Generated code:\n", code)

        code = code.strip() + "\n"
        local_vars = {"df": df.copy()}
        exec(code, {}, local_vars)

        result = local_vars.get("result", None)
        if result is None:
            return "✅ Code executed but no result was returned."

        summary_prompt = f"""
You are a helpful data analyst. The user asked:
"{question}"

Here is the result from the dataset:
{str(result)}

Summarize the findings clearly and concisely for the user.
"""

        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": summary_prompt}
            ],
            temperature=0.3
        )
        return summary_response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error executing generated code: {e}"
