import os
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai_openai import OpenAI
from pandasai_google import GoogleGemini
from pandasai.responses.response_parser import ResponseParser
from dotenv import load_dotenv

# Fungsi untuk mencatat error ke session state
def log_error(message):
    st.session_state.error_logs.append(message)
    st.warning(message)

# ----------------------------------------------------------------------------- 
# Custom Response Parser for Streamlit 
# ----------------------------------------------------------------------------- 
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        """Tampilkan dataframe dan simpan pesan placeholder ke cache.""" 
        st.dataframe(result["value"]) 
        st.session_state.answer_cache.append("[Displayed DataFrame]") 
        return

    def format_plot(self, result):
        """Tampilkan plot dan simpan pesan placeholder ke cache.""" 
        st.image(result["value"]) 
        st.session_state.answer_cache.append("[Displayed Plot]") 
        return

    def format_other(self, result):
        """Tampilkan hasil lain sebagai teks dan simpan ke cache.""" 
        st.write(str(result["value"])) 
        st.session_state.answer_cache.append(str(result["value"])) 
        return

# ----------------------------------------------------------------------------- 
# Validate Database Connection and Load Tables 
# ----------------------------------------------------------------------------- 
def validate_and_connect_database(credentials, llm):
    try:
        db_user = credentials["DB_USER"]
        db_password = credentials["DB_PASSWORD"]
        db_host = credentials["DB_HOST"]
        db_port = credentials["DB_PORT"]
        db_name = credentials["DB_NAME"]

        encoded_password = db_password.replace('@', '%40')
        engine = create_engine(
            f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        )

        with engine.connect() as connection:
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables_views = tables + views

            sdf_list = []
            table_info = {}

            for table in all_tables_views:
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm, "response_parser": StreamlitResponse(st)}
                    sdf_list.append(sdf)
                    table_info[table] = {
                        "columns": list(df.columns),
                        "row_count": len(df)
                    }
                except Exception as e:
                    err_msg = f"Failed to load data from public.{table}: {e}"
                    log_error(err_msg)
            datalake = SmartDatalake(sdf_list, config={"llm": llm, "response_parser": StreamlitResponse})
            return datalake, table_info, engine

    except Exception as e:
        log_error(f"Database connection error: {e}")
        return None, None, None

# ----------------------------------------------------------------------------- 
# Main function for Streamlit app 
# ----------------------------------------------------------------------------- 
def main():
    st.set_page_config(page_title="AI Database Explorer", layout="wide")
    st.title("🔍 AI Database Explorer")

    if "database_loaded" not in st.session_state:
        st.session_state.database_loaded = False
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = []
    if "error_logs" not in st.session_state:
        st.session_state.error_logs = []

    st.header("🔐 LLM Selection")
    llm_choice = st.selectbox("Select LLM", ["GroqCloud", "OpenAI", "Google Gemini"])
    api_key = st.text_input("API Key", type="password")

    if llm_choice == "GroqCloud":
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)
    elif llm_choice == "OpenAI":
        llm = OpenAI(api_key=api_key)
    elif llm_choice == "Google Gemini":
        llm = GoogleGemini(model_name="gemini-2.0-flash-thinking-exp-01-21", api_key=api_key)

    connect_button = st.button("Connect to Database")
    if connect_button and all([api_key]):
        with st.spinner("Connecting to the database and loading tables..."):
            credentials = {
                "DB_USER": os.getenv("DB_USER"),
                "DB_PASSWORD": os.getenv("DB_PASSWORD"),
                "DB_HOST": os.getenv("DB_HOST"),
                "DB_PORT": os.getenv("DB_PORT"),
                "DB_NAME": os.getenv("DB_NAME"),
            }
            datalake, table_info, engine = validate_and_connect_database(credentials, llm)

            if datalake and table_info:
                st.session_state.datalake = datalake
                st.session_state.table_info = table_info
                st.session_state.database_loaded = True
                st.session_state.error_logs.clear()
                st.success("Koneksi dan loading tabel selesai.")

    if st.session_state.database_loaded:
        st.header("💬 Query Data dengan Natural Language")
        with st.form(key="query_form"):
            prompt = st.text_input("Masukkan query Anda:")
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.answer_cache.clear()
                with st.spinner("Generating output..."):
                    try:
                        answer = st.session_state.datalake.chat(prompt)
                        st.session_state.answer_cache.append(answer)
                    except Exception as e:
                        st.error(f"Error processing query: {e}")

        if st.session_state.answer_cache:
            st.subheader("Output")
            for ans in st.session_state.answer_cache:
                st.write(ans)

if __name__ == "__main__":
    main()
