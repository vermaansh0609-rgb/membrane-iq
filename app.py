import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="MembraneIQ - Digital Twin", layout="wide")

st.title("MembraneIQ: Dynamic Water Purifier Prognostics & Virtual Sensing")
st.markdown("**Track 2: Water Purifier** | Physics-guided digital twin estimating membrane fouling and remaining useful life.")

# Sidebar Controls
st.sidebar.header("Input Water Parameters")
water_type = st.sidebar.selectbox("Feed Water Source", ["Municipal Tap", "Mixed Tanker Supply", "Borewell"])
daily_usage_liters = st.sidebar.slider("Daily Consumption (Liters)", min_value=5, max_value=40, value=15)
turbidity_spike = st.sidebar.checkbox("Simulate Heavy Silt / Turbidity Event")

# Physics Constants (Darcy Formulation)
clean_membrane_res = 1.0e11
viscosity = 0.001
differential_pressure = 400000.0  # 4 bar in Pa
critical_fouling_threshold = 4.5e11

# Data Generator Engine
def generate_simulation_data(water_source, usage, spike):
    days = np.arange(1, 181)
    fouling_rates = {
        "Municipal Tap": 2.0e7,
        "Mixed Tanker Supply": 3.4e7,
        "Borewell": 4.8e7
    }
    base_rate = fouling_rates[water_source]
    np.random.seed(42)
    noise = np.random.normal(0, 0.015e11, len(days))
    fouling_growth = (days * (usage / 15.0) * base_rate * 30) + noise
    
    if spike:
        fouling_growth[70:] += 1.1e11
        
    dynamic_resistance = clean_membrane_res + np.maximum(fouling_growth, 0)
    
    # Virtual Flow Rate calculation (standard domestic RO: 0.15 - 0.40 LPM)
    flow_rate = (differential_pressure / (viscosity * dynamic_resistance)) * 0.9e-4 * 60
    
    return pd.DataFrame({
        "Day": days,
        "Dynamic_Resistance": dynamic_resistance,
        "Flow_Rate_LPM": flow_rate,
        "Cumulative_Liters": days * usage
    })

df = generate_simulation_data(water_type, daily_usage_liters, turbidity_spike)

# Prognostic Regression
current_day = st.slider("Simulated Operational Day", min_value=10, max_value=170, value=45)
observed = df[df["Day"] <= current_day]

reg = LinearRegression()
reg.fit(observed[["Day"]], observed["Dynamic_Resistance"])
slope = reg.coef_[0]
intercept = reg.intercept_

if slope > 0:
    predicted_failure_day = (critical_fouling_threshold - intercept) / slope
    remaining_days = max(0, int(predicted_failure_day - current_day))
else:
    remaining_days = 365

current_res = observed["Dynamic_Resistance"].iloc[-1]
current_flow = observed["Flow_Rate_LPM"].iloc[-1]
health_index = max(0.0, min(100.0, 100.0 * (1.0 - (current_res - clean_membrane_res) / (critical_fouling_threshold - clean_membrane_res))))

# Metrics Display
c1, c2, c3, c4 = st.columns(4)
c1.metric("Membrane Health Index", f"{health_index:.1f} %")
c2.metric("Virtual Permeate Flow", f"{current_flow:.2f} LPM")
c3.metric("Projected Safe Lifespan", f"{remaining_days} Days")

if remaining_days <= 10 or health_index < 15.0:
    c4.error("CRITICAL: REPLACE FILTER")
elif remaining_days <= 25:
    c4.warning("MAINTENANCE ADVISORY")
else:
    c4.success("SYSTEM NORMAL")

# Visualizations
col_l, col_r = st.columns(2)
with col_l:
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=observed["Day"], y=observed["Dynamic_Resistance"], mode="lines", name="Observed Resistance", line=dict(color="#29b5e8", width=2)))
    future_days = np.arange(current_day, min(current_day + remaining_days + 20, 200))
    fig_res.add_trace(go.Scatter(x=future_days, y=intercept + slope * future_days, mode="lines", name="Prognostic Projection", line=dict(color="#ffaa00", dash="dash")))
    fig_res.add_hline(y=critical_fouling_threshold, line_dash="dot", line_color="#ff4b4b", annotation_text="Breakthrough Limit")
    fig_res.update_layout(title="Hydraulic Resistance vs. Time", xaxis_title="Days", yaxis_title="Resistance (m⁻¹)", height=380)
    st.plotly_chart(fig_res, use_container_width=True)

with col_r:
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(x=observed["Day"], y=observed["Flow_Rate_LPM"], mode="lines", name="Virtual Permeate Flow", line=dict(color="#00d26a", width=2)))
    fig_flow.add_hline(y=0.15, line_dash="dot", line_color="#ff4b4b", annotation_text="Dry-Run Shutoff Cutoff (0.15 LPM)")
    fig_flow.update_layout(title="Flow Decay Curve", xaxis_title="Days", yaxis_title="Flow Rate (LPM)", height=380)
    st.plotly_chart(fig_flow, use_container_width=True)

# Telemetry Log & Safety Cutoff
st.subheader("Telemetry & Anomaly Safety Log")
display_log = observed.tail(5)[["Day", "Cumulative_Liters", "Flow_Rate_LPM", "Dynamic_Resistance"]].copy()
display_log["Failsafe_Status"] = "NORMAL RUN"
display_log.loc[display_log["Flow_Rate_LPM"] < 0.18, "Failsafe_Status"] = "ENGAGED (PROTECTION)"
st.dataframe(display_log, use_container_width=True)
