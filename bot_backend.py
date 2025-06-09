import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Data dictionary mapping column names to descriptions
data_dictionary = {
    "Campaign": "The marketing campaign name. This is lengthy and will contain codes and other alpha numerics",
    "Clicks": "Number of times the ad was clicked",
    "Impressions": "Number of times the ad was shown",
    "Media Cost": "Total ad spend in USD",
    "Date": "The date the ad was served (YYYY-MM-DD)",
    "Total Conversions": "Number of desired outcomes (e.g., signups, purchases)",
    "advertiser": "The client that this table belongs to. Will be the same value",
    "Activity ID": "Unique identifier for certain activities. Not all activities will have a set ID. Will not be important for analysis",
    "Placement": "Where this ad was placed. This is a very specific and long alpha numeric that contains where the ad was placed, what type of ad was placed, and even the layout size of the ad",
    "Video Plays": "The amount of times a video was played for that ad",
    "video_completions": "The amount of times a user has completed a ad video in its entirety"
}

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    sample_df = df.sample(n=min(3, len(df)), random_state=42)

    dictionary_text = "\n".join([
        f"{col}: {desc}" for col, desc in data_dictionary.items()
    ])

    planning_prompt = f"""
You are a Python data analyst.

You will be given:
1. A user question
2. A data dictionary
3. A small preview of the dataset

Your job is to write Python code that uses the `df` DataFrame to compute intermediate variables, summaries, or insights that help answer the user's question. 
Use only the columns described in the data dictionary. You can create multiple variables if needed.

Do not wrap in triple backticks. Do not explain. Just output the Python code.

User question:
{question}

Data dictionary:
{dictionary_text}

Sample data:
{sample_df.to_string(index=False)}
"""

    try:
        code_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": planning_prompt}
            ],
            temperature=0.3
        )
        code = code_response.choices[0].message.content.strip()

        print("\U0001F4CA Generated code:\n", code)

        local_vars = {"df": df}
        exec(code, {}, local_vars)

        result_vars = {k: v for k, v in local_vars.items() if k not in ["df", "__builtins__"]}
        result_summary = "\n\n".join(
            f"{k} =\n{v.to_markdown(index=False) if isinstance(v, pd.DataFrame) else v}"
            for k, v in result_vars.items()
        )

        # Explanation phase
        explain_prompt = f"""
The user originally asked: "{question}"

The following Python code was executed to analyze the data:

{code}

The resulting variables are:

{result_summary}

Please write a helpful answer to the user explaining the results in plain English.
"""

        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": explain_prompt}
            ],
            temperature=0.3
        )

        return f"**Answer:** {summary_response.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ Error: {e}"
