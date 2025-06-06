import streamlit as st
from bot_backend import analyze_question

import streamlit as st
#st.write("✅ Secrets loaded:", "OPENAI_API_KEY" in st.secrets)

st.title("Looker Chatbot!")

sheet_url = st.text_input("Paste the Google Sheet URL:")
question = st.text_area("Ask a question about this community:")

if st.button("Ask") and sheet_url and question:
    with st.spinner("Thinking..."):
        response = analyze_question(question, sheet_url)
        st.markdown(f"**Answer:** {response}")
# trigger redeploy
