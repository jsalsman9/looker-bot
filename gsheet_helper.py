import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import streamlit as st

def load_sheet_data(sheet_url: str) -> pd.DataFrame:
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)

        sheet = client.open_by_url(sheet_url).sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Failed to load sheet: {e}")
        return pd.DataFrame()
