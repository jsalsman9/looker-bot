import streamlit as st
from bot_backend import analyze_question

# Map of community names to Google Sheet URLs
COMMUNITY_SHEETS = {
    "Maplewood": "https://docs.google.com/spreadsheets/d/1zqONqW7dkKMhjT3VBWMYklar4EzE4PpCG38H9nqqhMc/edit?gid=0#gid=0",
    "Kingswood": "https://docs.google.com/spreadsheets/d/your-oakridge-sheet-id",
    "Duncaster": "https://docs.google.com/spreadsheets/d/your-pinehill-sheet-id",
}

st.title("📊 Campaign Performance Bot")

community = st.selectbox("Select a community", list(COMMUNITY_SHEETS.keys()))
question = st.text_input("Ask a question about the campaign data:")

if st.button("Ask") and question:
    sheet_url = COMMUNITY_SHEETS[community]
    with st.spinner("Thinking..."):
        response = analyze_question(question, sheet_url)
    st.markdown(response)

