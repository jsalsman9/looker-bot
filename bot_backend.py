import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from gsheet_helper import load_sheet_data
import json
import traceback

load_dotenv()
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# You can extend this as needed
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
When asked about performance, consider metrics like:
- Click-through rate (CTR = Clicks / Impressions)
- Cost per acquisition (CPA = Media Cost / Conversions, lower is better)
- Conversion rate = Conversions / Clicks
- Return on ad spend (ROAS = Revenue / Media Cost)
"""

def analyze_question(question: str, sheet_url: str):
    df = load_sheet_data(sheet_url)
    if df.empty:
        return "❌ Could not load data from the sheet."

    # Sample and prepare dictionary text
    sample_df = df.sample(n=min(3, len(df)), random_state=42)
    dictionary_text = "\n".join([f"{col}: {desc}" for col, desc in data_dictionary.items() if col in df.columns])

    # Step 1: Planning prompt to generate code
    planning_prompt = f"""
You are a Python data analyst bot.

Your job is to write pure Python code (no explanation) that will answer the user's question.
You are given:
- A data dictionary
- A few rows of example data (in a pandas DataFrame called df)
- A guide to useful metrics
- The user's question

✅ The df variable is already loaded.
✅ You may create multiple result variables if useful.
✅ Only use columns that exist in the data dictionary or preview.

❌ Do NOT import anything.
❌ Do NOT define the DataFrame or read files.
❌ Do NOT wrap the code in markdown or quotes.
❌ Do NOT include explanations.

📌 Your job is to return **just the Python code** that performs any necessary filtering, aggregating, or deriving metrics to answer the user's question. Use pandas idioms.

User question: {question}

Data dictionary:
{dictionary_text}

Example data:
{sample_df.to_markdown(index=False)}

{kpi_guide}
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
        print("🔧 Generated code:\n", generated_code)

        # Step 2: Execute code safely in local scope
        local_vars = {"df": df.copy()}
        exec(generated_code, {}, local_vars)

        # Step 3: Gather all result variables (non-dataframe summaries)
        outputs = {}
        for name, val in local_vars.items():
            if name == "df":
                continue
            if isinstance(val, pd.DataFrame):
                outputs[name] = val.to_markdown(index=False)
            else:
                outputs[name] = str(val)

        if not outputs:
            return "✅ Code ran but no result variables were returned."

        # Step 4: Summarize results with GPT
        output_text = "\n\n".join([f"{k}:\n{v}" for k, v in outputs.items()])
        summary_prompt = f"""
You are a helpful data analyst.

The user asked:
{question}

You ran the analysis and got the following results:
{output_text}

Write a short, clear answer to the user's question using the results above.
"""
        summary_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": summary_prompt}
            ],
            temperature=0.3
        )

        return f"**Answer:** {summary_response.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
