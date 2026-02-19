import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad
from datetime import datetime, timedelta

# --- Path Setup ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Import your custom modules
try:
    from src.data_sources.options_openbb import get_option_chains
    from src.modelling.minimization import mixture_pdf
    from scripts.run_build_pdf import return_optimal_distribution_parameters
except ImportError as e:
    # Fallback for demonstration if local modules are missing
    st.error(f"Module Import Error: {e}. Ensure you are running from the project root.")
    # Dummy functions to allow UI testing without backend
    def get_option_chains(ticker): return pd.DataFrame({'dte': [7, 30, 60], 'underlying_price': [150.0]*3})
    def return_optimal_distribution_parameters(t, d): return (0.0, 0.1, 0.0, 0.2, 0.5) # Dummy params
    def mixture_pdf(x, a1, b1, a2, b2, q): return 1/(x*0.1*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-5)**2/(2*0.1**2)) # Dummy LogNormal

st.set_page_config(layout='centered', page_title="Options Implied PDF Analyzer")

# --- Helper Functions ---
def get_expiry_date(days_to_expiry):
    """Calculates target date from DTE."""
    target_date = datetime.now() + timedelta(days=int(days_to_expiry))
    return target_date.strftime("%Y-%m-%d")

# --- 1. Header & Data Ingestion Section ---
st.title('Options-Implied Price Distribution')
st.header('A WebApp to Analyze Implied Future Price Distributions from Option Chains')
st.markdown("""
This application is based on the minimization framework outlined in the paper 'Probability distributions of future asset prices implied by
option prices' by Bhupinder Bahra.
            
To see the methodology and code details, please refer to the [GitHub Repository](https://github.com/aatj887/call-implied-price-distribution-framework).
""")

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input('Ticker Symbol', label_visibility="collapsed", placeholder="Enter Ticker (e.g. SPY)")
with col2:
    fetch_btn = st.button("⬇ Download Data", use_container_width=True)

if fetch_btn:
    st.cache_data.clear()
    try:
        with st.spinner(f"Fetching Option Chain for {ticker_input}..."):
            df = get_option_chains(ticker_input.upper())
            st.session_state['chain_data'] = df
            st.session_state['ticker'] = ticker_input.upper()
            # Clear downstream states
            keys_to_drop = ['params', 'selected_dte', 'spot']
            for k in keys_to_drop:
                if k in st.session_state: del st.session_state[k]
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")

# --- 2. Expiry Selection & Model Fitting ---
if 'chain_data' in st.session_state:
    df = st.session_state['chain_data']
    try:
        current_spot = float(df['underlying_price'].iloc[0])
        st.session_state['spot'] = current_spot
    except:
        current_spot = 100.0 # Fallback

    st.divider()
    st.subheader(f"1. Select Expiry of Options for {st.session_state['ticker']} (Current Price: {current_spot:.2f})")
    
    # Create friendly labels for the dropdown (DTE + Date)
    unique_dtes = sorted(df['dte'].unique())
    dte_options = {dte: f"{dte} Days ({get_expiry_date(dte)})" for dte in unique_dtes}
    
    selected_dte = st.selectbox(
        "Choose Expiration:", 
        options=unique_dtes, 
        format_func=lambda x: dte_options[x]
    )

    # Fit Model Button
    if st.button("Calculate Optimal Distribution Parameters"):
        with st.spinner('Optimizing Mixture Model...'):
            try:
                # Run optimization
                params = return_optimal_distribution_parameters(st.session_state['ticker'], selected_dte)
                st.session_state['params'] = params
                st.session_state['selected_dte'] = selected_dte
                st.success("Model Fitted Successfully")
            except Exception as e:
                st.error(f"Optimization failed: {e}")

# --- 3. Probability Analysis & Visualization ---
if 'params' in st.session_state:
    a1, b1, a2, b2, q = st.session_state['params']
    spot = st.session_state['spot']
    
    st.divider()
    st.subheader("2. Analyze Probabilities")
    
    # --- Input Configuration ---
    c_mode, c_range = st.columns(2)
    with c_mode:
        input_mode = st.radio("Input Unit:", ["Price ($)", "Return (%)"], horizontal=True)
    with c_range:
        range_type = st.radio("Target Range:", ["Below X", "Between X and Y", "Above X"], horizontal=True)

    # --- Dynamic Inputs based on selection ---
    low_limit, high_limit = 0.0, np.inf
    
    # Helper to convert Return to Price
    def ret_to_price(r): return spot * (1 + r/100)
    def price_to_ret(p): return ((p / spot) - 1) * 100

    col_input = st.container()
    
    with col_input:
        if range_type == "Below X":
            if input_mode == "Price ($)":
                val = st.number_input("Price X ($)", value=float(round(spot*0.9, 2)))
                high_limit = val
            else:
                val = st.number_input("Return X (%)", value=-10.0)
                high_limit = ret_to_price(val)
                
        elif range_type == "Between X and Y":
            c1, c2 = st.columns(2)
            if input_mode == "Price ($)":
                with c1: val_x = st.number_input("Lower Price X ($)", value=float(round(spot*0.95, 2)))
                with c2: val_y = st.number_input("Upper Price Y ($)", value=float(round(spot*1.05, 2)))
                low_limit, high_limit = val_x, val_y
            else:
                with c1: val_x = st.number_input("Lower Return X (%)", value=-5.0)
                with c2: val_y = st.number_input("Upper Return Y (%)", value=5.0)
                low_limit, high_limit = ret_to_price(val_x), ret_to_price(val_y)
                
        elif range_type == "Above X":
            if input_mode == "Price ($)":
                val = st.number_input("Price X ($)", value=float(round(spot*1.1, 2)))
                low_limit = val
            else:
                val = st.number_input("Return X (%)", value=10.0)
                low_limit = ret_to_price(val)

    # --- Calculations ---
    # Define integration bounds (handle infinity for calculations)
    calc_low = max(0.0001, low_limit) # Prevent 0 division in log models
    calc_high = high_limit if high_limit != np.inf else spot * 20 # Arbitrary high cap for numerical integration
    
    prob_val, _ = quad(mixture_pdf, calc_low, calc_high, args=(a1, b1, a2, b2, q))
    
    # Validate Total Area (Model Health Check)
    total_area, _ = quad(mixture_pdf, 0.0001, spot * 50, args=(a1, b1, a2, b2, q))
    
    # Display Metric
    st.metric(
        label=f"Probability ({range_type})", 
        value=f"{prob_val*100:.4f}%",
        delta=f"Model Error: ±{abs(1.0 - total_area)*100:.2e} %",
        delta_color="inverse"
    )
    
    if abs(1.0 - total_area) > 0.05:
        st.warning("⚠️ The area under the curve is deviating significantly from 1.0. Model parameters may be unstable.")

    # --- Visualization Logic ---

    # 1. Define the Price range for the underlying math
    s_plot = np.linspace(0.01, spot * 2.5, 1000) 
    pdf_plot = np.array([mixture_pdf(s, a1, b1, a2, b2, q) for s in s_plot])

    # 2. Determine X-Axis values based on the toggle
    if input_mode == "Price ($)":
        x_vals = s_plot
        x_label = "Stock Price ($)"
        v_line_pos = spot
        # Boundary logic for shading
        mask_low = low_limit
        mask_high = high_limit
    else:
        # Convert Price array to Return array: ((P/Spot) - 1) * 100
        x_vals = ((s_plot / spot) - 1) * 100
        x_label = "Return (%)"
        v_line_pos = 0.0 # Spot price is 0% return
        # Convert price bounds back to return bounds for the mask
        mask_low = ((low_limit / spot) - 1) * 100
        mask_high = ((high_limit / spot) - 1) * 100

    # 3. Create the Mask for shading
    mask = (x_vals >= mask_low) & (x_vals <= (mask_high if high_limit != np.inf else max(x_vals)))

    # 4. Build the Plotly Figure
    fig = go.Figure()

    # Main PDF Curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=pdf_plot, 
        mode='lines', name='Implied PDF', 
        line=dict(color='#1f77b4', width=2)
    ))

    # Shaded Area
    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=x_vals[mask], y=pdf_plot[mask], 
            fill='tozeroy', mode='none', name='Selected Range', 
            fillcolor='rgba(31, 119, 180, 0.4)'
        ))

    # Vertical Line for "At-The-Money" (Spot or 0%)
    fig.add_vline(x=v_line_pos, line_dash="dash", line_color="orange", 
                annotation_text="Current Spot" if input_mode == "Price ($)" else "At-the-Money")

    fig.update_layout(
        title=f"Implied Probability Density ({st.session_state['ticker']})",
        xaxis_title=x_label,
        yaxis_title="Density",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)