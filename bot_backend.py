import os
import json
import streamlit as st
import pandas as pd
import numpy as np
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

    # Small sample for GPT context
    sample_df = df.sample(n=min(3, len(df)), random_state=42)

    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items()])

    planning_prompt = f"""
You are a Python data analyst. Your job is to write Python code that answers the user's question using ONLY the available data.

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_string(index=False)}

User question: "{question}"

Please return a Python code block that:
- Uses only the data in the DataFrame `df`
- Defines one or more result variables (e.g., `result`, `top_campaigns`, etc.)
- Avoids any placeholder or imaginary variables
- Always wraps string values in quotes
- Does NOT include print() or display() statements
- Does NOT include import statements (those are already handled)
- Handles edge cases like division by zero or missing values

✅ Output ONLY a complete Python code block, no explanations.
"""

    plan_response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": planning_prompt}
        ],
        temperature=0.2
    )

    generated_code = plan_response.choices[0].message.content.strip()
    if generated_code.startswith("```"):
        generated_code = generated_code.strip("`\n")
        if generated_code.startswith("python"):
            generated_code = generated_code[6:].strip()

    local_vars = {"df": df, "pd": pd, "np": np}

    try:
        print("\n📄 Generated Code:\n", generated_code)
        exec(generated_code, {}, local_vars)
    except Exception as e:
        return f"❌ Error executing generated code: {e}\n\n📄 Code was:\n{generated_code}"

    result_keys = [k for k in local_vars if not k.startswith("__") and k != "df"]
    if not result_keys:
        return "✅ Code executed, but no result objects were defined."

    try:
        explanation_prompt = f"""
You are a helpful digital marketing data analyst. The user asked:
"{question}"

Here is the result of your analysis:
{str({k: local_vars[k] for k in result_keys})}

Please explain your findings in a short, clear, and professional way.
"""

        summary = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "system", "content": explanation_prompt}],
            temperature=0.4
        )
        return f" {summary.choices[0].message.content.strip()}"

    except Exception as e:
        return f"✅ Executed code but failed to summarize: {e}\n\n{str({k: local_vars[k] for k in result_keys})}"
