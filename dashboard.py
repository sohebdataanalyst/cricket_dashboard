import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Cricket Analytics", layout="wide")

st.title("🏏 Live Cricket Analytics & Score Prediction")
st.markdown("### Developed by: Soheb Khan")

# --- STEP 1: DATA HANDLING ---
def load_data():
    try:
        df_temp = pd.read_csv('cricket_scores.csv')
        if 'runs_in_over' not in df_temp.columns:
            raise KeyError
        return df_temp
    except:
        data = {
            'over': list(range(1, 21)),
            'runs_in_over': [8, 12, 6, 10, 15, 7, 9, 11, 5, 14, 8, 10, 12, 6, 18, 9, 11, 7, 13, 10],
            'wickets_in_over': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1]
        }
        return pd.DataFrame(data)

df = load_data()
df['cumulative_runs'] = df['runs_in_over'].cumsum()
df['cumulative_wickets'] = df['wickets_in_over'].cumsum()

# --- STEP 2: SIDEBAR FILTERS & INPUTS ---
st.sidebar.header("Match Settings & Filters")
# TEAM FORMAT FILTER
format_type = st.sidebar.selectbox("Select Match Format", ["T20", "ODI", "Test"])
total_overs = 20 if format_type == "T20" else (50 if format_type == "ODI" else 90)

st.sidebar.divider()
st.sidebar.header("Live Controls")
curr_over = st.sidebar.slider("Current Over", 1, total_overs, 8)
curr_runs = st.sidebar.number_input("Current Score", value=int(df['cumulative_runs'].iloc[curr_over-1] if curr_over <= len(df) else 100))
curr_wickets = st.sidebar.number_input("Wickets", 0, 10, value=int(df['cumulative_wickets'].iloc[curr_over-1] if curr_over <= len(df) else 2))
target_score = st.sidebar.number_input("Target Score", value=180)

# --- STEP 3: ANALYTICS LOGIC ---
# Linear Regression for Prediction
X = np.array(list(range(1, len(df)+1))).reshape(-1, 1)
y = df['cumulative_runs'].values
model = LinearRegression().fit(X, y)
predicted_final = int(model.predict([[total_overs]])[0])

# Win Probability (Logic based on RRR and Wickets)
rrr = (target_score - curr_runs) / (total_overs - curr_over) if total_overs > curr_over else 0
win_prob = max(5, min(95, 100 - (rrr * 5) - (curr_wickets * 7))) # Rough estimation logic

# --- STEP 4: TOP METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Score", f"{curr_runs}/{curr_wickets}")
m2.metric("Required RR", round(rrr, 2))
m3.metric("Projected Score", predicted_final)
m4.metric("Win Probability", f"{int(win_prob)}%")

st.divider()

# --- STEP 5: GRAPHS ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    # 1. Innings Progress
    fig1 = px.line(df[:curr_over], x='over', y='cumulative_runs', title="Innings Progress", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    # 2. Win Probability Gauge (NEW)
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = win_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Win Probability %"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ))
    st.plotly_chart(fig2, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # 3. Required vs Current Run Rate (NEW)
    crr = curr_runs / curr_over
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=['Current RR', 'Required RR'], y=[crr, rrr], marker_color=['blue', 'red']))
    fig3.update_layout(title="Run Rate Comparison")
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    # 4. Scoring Breakdown
    fig4 = px.pie(names=['Boundaries', 'Singles', 'Extras'], 
                 values=[curr_runs*0.55, curr_runs*0.35, curr_runs*0.1], title="Scoring Breakdown")
    st.plotly_chart(fig4, use_container_width=True)

st.success(f"✅ Dashboard updated for {format_type} format!")