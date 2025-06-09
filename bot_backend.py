import os
import streamlit as st
import pandas as pd
import json
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
    dictionary_text = "\n".join([f"{k}: {v}" for k, v in data_dictionary.items() if k in df.columns])

    # Step 1: Generate analysis code
    planning_prompt = f"""
You are a Python data analyst bot. You will receive:
1. A user question
2. A data dictionary
3. A sample of the data

Write Python code using only available columns in the DataFrame `df` to answer the user's question.
The final result should be stored in a variable named `result`. This can be a DataFrame or a string.
Don't include explanations, comments, or markdown. Just clean Python code.

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_string(index=False)}

User question:
"{question}"
"""

    code_response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": planning_prompt}
        ],
        temperature=0.3
    )

    generated_code = code_response.choices[0].message.content.strip()
    st.code(generated_code, language="python")  # Optional: show the generated code

    # Step 2: Execute safely
    local_vars = {"df": df}
    try:
        exec(generated_code, {}, local_vars)
        result = local_vars.get("result", None)
        if result is None:
            return "❌ Code did not produce a variable named `result`."
    except Exception as e:
        return f"❌ Error executing generated code: {e}"

    # Step 3: Summarize result
    result_preview = result.to_string(index=False) if isinstance(result, pd.DataFrame) else str(result)
    summary_prompt = f"""
You are a helpful data analyst. A user asked:
"{question}"

Here is the output from a data analysis:
{result_preview}

Write a short, clear answer to the user's question.
"""

    try:
        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": summary_prompt}
            ],
            temperature=0.3
        )
        return f"**Answer:** {summary_response.choices[0].message.content.strip()}"
    except Exception as e:
        return f"✅ Code ran successfully but failed to summarize: {e}\n\n{result_preview}"
