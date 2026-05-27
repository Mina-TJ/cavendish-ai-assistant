# Cavendish Farms AI Assistant
# Goal: VP can ask questions in plain English and get instant answers
# Method: OpenAI GPT-4o mini + pandas queries on real dataset
# Deployed on Streamlit Cloud - accessible from any browser

import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# Initialize OpenAI client using Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Cavendish Farms colors
CF_GREEN = "#1A4731"
CF_GOLD = "#F9A825"

# Page configuration
st.set_page_config(
    page_title="Cavendish Farms AI Assistant",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    .main-header {{
        background-color: {CF_GREEN};
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .main-header h1 {{ color: {CF_GOLD}; font-size: 28px; margin: 0; }}
    .main-header p {{ color: #FFFFFF; margin: 5px 0 0 0; font-size: 14px; }}
    .insight-box {{
        background-color: #F5F5F5;
        padding: 15px;
        border-left: 4px solid {CF_GREEN};
        border-radius: 5px;
        margin-bottom: 10px;
    }}
    .stButton > button {{
        background-color: {CF_GREEN};
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: {CF_GOLD};
        color: {CF_GREEN};
    }}
    </style>
""", unsafe_allow_html=True)

# Password protection
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown(f"""
            <div style="background-color:{CF_GREEN};
            padding:20px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:{CF_GOLD};margin:0">
            Cavendish Farms — Supply Chain AI Assistant</h2>
            <p style="color:white;margin:5px 0 0 0">
            Please enter the access password to continue</p>
            </div>
        """, unsafe_allow_html=True)

        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        return False
    return True

if not check_password():
    st.stop()

# Header
st.markdown(f"""
    <div class="main-header">
        <h1>Cavendish Farms — Supply Chain AI Assistant</h1>
        <p>Ask any question about transport loads in plain English</p>
    </div>
""", unsafe_allow_html=True)

# Load datasets
@st.cache_data
def load_data():
    df_transport = pd.read_csv("Transportation data_test.csv")
    df_customer = pd.read_excel("Customer Group_Test.xlsx")
    df_transport["Create Date"] = pd.to_datetime(df_transport["Create Date"])
    df_transport["Status_Label"] = df_transport["Status"].map(
        {"Booked": "Covered", "Tendered": "Uncovered"}
    )
    return df_transport, df_customer

df_transport, df_customer = load_data()

# Build data summary for AI context
def get_data_summary():
    total = len(df_transport)
    covered = len(df_transport[df_transport["Status"] == "Booked"])
    uncovered = len(df_transport[df_transport["Status"] == "Tendered"])

    uncovered_by_state = (
        df_transport[df_transport["Status"] == "Tendered"]
        .groupby("Dest State").size()
        .sort_values(ascending=False)
        .head(10).to_dict()
    )

    covered_by_state = (
        df_transport[df_transport["Status"] == "Booked"]
        .groupby("Dest State").size()
        .sort_values(ascending=False)
        .head(10).to_dict()
    )

    country_all = df_transport.groupby("Dest Ctry").size().to_dict()
    country_covered = df_transport[
        df_transport["Status"] == "Booked"
    ].groupby("Dest Ctry").size().to_dict()
    country_uncovered = df_transport[
        df_transport["Status"] == "Tendered"
    ].groupby("Dest Ctry").size().to_dict()

    df_merged = df_transport.merge(
        df_customer[["SHIP TO NUMBER", "SALES DIVISION NAME"]],
        left_on="Dest Code",
        right_on="SHIP TO NUMBER",
        how="left"
    )
    uncovered_by_division = (
        df_merged[df_merged["Status"] == "Tendered"]
        .groupby("SALES DIVISION NAME").size()
        .sort_values(ascending=False)
        .to_dict()
    )
    total_by_division = (
        df_merged.groupby("SALES DIVISION NAME").size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # Load risk scores from data folder
    risk_df = pd.read_csv("risk_scores.csv")
    risk_data = risk_df[[
        "Dest State", "Uncovered_Loads",
        "Total_Loads", "Risk_Score", "Risk_Level"
    ]].to_dict(orient="records")

    # Load forecast from data folder
    forecast_df = pd.read_csv("forecast_uncovered.csv")
    forecast_data = forecast_df[[
        "State/Province", "Forecast Next 14 Days"
    ]].to_dict(orient="records")

    summary = f"""
    You are an AI assistant for Cavendish Farms supply chain team.
    You have access to transport load data with the following key facts:

    DATASET OVERVIEW:
    - Total transport loads: {total}
    - Covered loads (Booked - carrier confirmed): {covered} ({round(covered/total*100,1)}%)
    - Uncovered loads (Tendered - no carrier yet): {uncovered} ({round(uncovered/total*100,1)}%)
    - Date range: January 27 to March 15, 2022
    - A load is UNCOVERED when it is Tendered (no carrier acceptance yet)
    - A load is COVERED when it is Booked (carrier confirmed)

    TOP 10 STATES BY UNCOVERED LOADS:
    {uncovered_by_state}

    TOP 10 STATES BY COVERED LOADS:
    {covered_by_state}

    LOADS BY COUNTRY (all loads combined):
    - US: {country_all.get('US', 0)} total ({country_covered.get('US', 0)} covered, {country_uncovered.get('US', 0)} uncovered) = {round(country_all.get('US', 0)/total*100, 1)}% of all loads
    - CA: {country_all.get('CA', 0)} total ({country_covered.get('CA', 0)} covered, {country_uncovered.get('CA', 0)} uncovered) = {round(country_all.get('CA', 0)/total*100, 1)}% of all loads

    SALES DIVISIONS - TOTAL AND UNCOVERED LOADS:
    Total loads by division: {total_by_division}
    Uncovered loads by division: {uncovered_by_division}
    Note: To prioritize divisions focus on which has the most uncovered loads

    RISK SCORES (uncovered loads only):
    {risk_data}

    FORECASTED UNCOVERED LOADS NEXT 14 DAYS:
    {forecast_data}

    ANOMALIES DETECTED:
    - Z-Score: NY flagged as anomalous in uncovered loads (Z=2.36)
    - Isolation Forest: 6 anomalous uncovered loads detected
    - Heaviest anomalous load: Ontario, 55180 lbs, Tendered

    CLUSTERING (uncovered loads - 9 route archetypes):
    - Largest uncovered clusters: PA via PLT3 (22 loads), NY via PLT3 (19 loads)
    - Most urgent: PA and NY routes need immediate carrier coverage

    Answer questions clearly and concisely in plain English.
    Always refer to Booked as Covered and Tendered as Uncovered.
    Focus on actionable insights for the VP of supply chain.
    If asked about something not in the data, respond exactly like this:
    "Sorry, I don't have that information in the current dataset.
    I can only answer questions about transport loads,
    covered/uncovered status, destinations, divisions,
    risk scores, and forecasts."
    """
    return summary

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Two column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Question")

    # Welcome message shown only at the start
    if len(st.session_state.messages) == 0:
        st.markdown(f"""
            <div style="background-color:#E8F5E9;
            padding:15px;border-radius:5px;
            margin:5px 0;border-left:4px solid {CF_GREEN}">
            <b>AI Assistant:</b> Hi! I am your Cavendish Farms Supply Chain AI Assistant.
            I can answer questions about transport loads, covered and uncovered status,
            destinations, divisions, risk scores, and forecasts.
            How can I help you today?
            </div>
        """, unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
                <div style="background-color:#E8F5E9;
                padding:10px;border-radius:5px;
                margin:5px 0;border-left:4px solid {CF_GREEN}">
                <b>You:</b> {msg["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#FFF8E1;
                padding:10px;border-radius:5px;
                margin:5px 0;border-left:4px solid {CF_GOLD}">
                <b>AI Assistant:</b> {msg["content"]}
                </div>
            """, unsafe_allow_html=True)

    # Question input
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g. Which states have the most uncovered loads?"
    )

    if st.button("Ask"):
        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            api_messages = [
                {"role": "system", "content": get_data_summary()}
            ]
            for msg in st.session_state.messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=api_messages,
                    max_tokens=500,
                    temperature=0.3
                )
                answer = response.choices[0].message.content

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    st.subheader("Key Insights")

    st.markdown(f"""
        <div class="insight-box">
        <b>Total Loads</b><br>
        837 transport loads<br>
        Jan 27 - Mar 15, 2022
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="insight-box">
        <b>Covered vs Uncovered</b><br>
        Covered (Booked): 724 (86.5%)<br>
        Uncovered (Tendered): 113 (13.5%)
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="insight-box">
        <b>Top Uncovered States</b><br>
        NY: 16 loads<br>
        PA: 14 loads<br>
        AB: 11 loads<br>
        ON: 11 loads
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="insight-box">
        <b>Highest Risk States</b><br>
        SC: Score 73.3 (High)<br>
        PA: Score 71.6 (High)<br>
        NY: Score 61.9 (High)
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="insight-box">
        <b>Sample Questions</b><br>
        - Which states have the most uncovered loads?<br>
        - Which division should we prioritize?<br>
        - What percentage of loads go to Canada?<br>
        - Which states are highest risk?<br>
        - What do you forecast for next 2 weeks?
        </div>
    """, unsafe_allow_html=True)