import openai
import os
from openai import OpenAI
import pandas as pd
from gsheet_helper import load_sheet_data
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]  # Use only secrets on Streamlit Cloud
client = OpenAI(api_key=api_key)


def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "Could not load data from the sheet."

    system_prompt = f"""
You are a helpful data analyst. You are working with this Google Sheets data:

{df.head(5).to_string(index=False)}

Only use this data. The user will ask questions about trends, outliers, summaries, etc.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

