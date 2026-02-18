import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad

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
    st.error(f"Module Import Error: {e}. Ensure you are running from the project root.")

st.set_page_config(layout='centered', page_title="Options Implied PDF Analyzer")
st.title('📊 Options Implied Price Distribution')

# --- 1. Cached Data Fetching ---
@st.cache_data
def load_data(ticker_sym):
    """Fetches option chain data and caches it to prevent repeated API calls."""
    return get_option_chains(ticker_sym)

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    
    # Cache Control
    if st.button("🔄 Clear Market Data Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

    st.divider()
    
    # Configuration Form
    with st.form("config_form"):
        ticker_input = st.text_input('Ticker Symbol', value='AAPL').upper()
        
        # Pre-fetch data to populate the DTE dropdown
        try:
            temp_data = load_data(ticker_input)
            available_expiries = sorted(temp_data['dte'].unique())
            underlying_price = float(temp_data['underlying_price'].iloc[0])
        except Exception as e:
            st.error(f"Error loading {ticker_input}: {e}")
            available_expiries = [0]
            underlying_price = 0.0

        dte_input = st.selectbox('Days to Expiry (DTE)', options=available_expiries, index=0)
        
        submit_calc = st.form_submit_button("Calculate Model Parameters")

# --- 3. Model Calculation ---
if submit_calc:
    with st.spinner(f'Fitting mixture model for {ticker_input}...'):
        # This function runs the optimization to find a1, b1, a2, b2, q
        params = return_optimal_distribution_parameters(ticker_input, dte_input)
        st.session_state['params'] = params
        st.session_state['ticker'] = ticker_input
        st.session_state['spot'] = underlying_price
        st.success(f"Model parameters optimized for {ticker_input}")

# --- 4. Probability Analysis Section ---
if 'params' in st.session_state:
    a1, b1, a2, b2, q = st.session_state['params']
    spot = st.session_state['spot']
    
    st.divider()
    st.header('Probability Calculator')
    st.info("Define a price range to calculate the exact probability of the stock finishing within those bounds.")

    # Form for price bounds to prevent constant reruns
    with st.form("probability_form"):
        col1, col2 = st.columns(2)
        with col1:
            low_b = st.number_input('Lower Price Bound ($)', value=round(spot * 0.95, 2))
        with col2:
            high_b = st.number_input('Upper Price Bound ($)', value=round(spot * 1.05, 2))
        
        submit_viz = st.form_submit_button('Update Analysis')

    if submit_viz or 'range' not in st.session_state:
        st.session_state['range'] = (low_b, high_b)

    # Execution of Math and Plotting
    if 'range' in st.session_state:
        low, high = st.session_state['range']
        
        # A. High-Precision Integration using scipy.integrate.quad
        # This integrates the mixture_pdf function directly
        prob_val, error = quad(mixture_pdf, low, high, args=(a1, b1, a2, b2, q))
        
        # B. Generate data for Plotly visualization
        s_plot = np.linspace(0, spot * 20, 1000)
        pdf_plot = np.array([mixture_pdf(s, a1, b1, a2, b2, q) for s in s_plot])
        
        # Mask for the shaded area
        mask = (s_plot >= low) & (s_plot <= high)
        s_shaded = s_plot[mask]
        pdf_shaded = pdf_plot[mask]

        # C. Display Results
        st.metric(
            label=f"Probability of Price between ${low:.2f} and ${high:.2f}", 
            value=f"{prob_val*100:.4f}%"
        )

        # D. Plotting
        fig = go.Figure()

        # Full Distribution
        fig.add_trace(go.Scatter(
            x=s_plot, y=pdf_plot, 
            mode='lines', name='Implied PDF', 
            line=dict(color='#1f77b4', width=2)
        ))

        # Shaded Selection
        fig.add_trace(go.Scatter(
            x=s_shaded, y=pdf_shaded, 
            fill='tozeroy', mode='none', 
            name='Target Range', 
            fillcolor='rgba(31, 119, 180, 0.4)'
        ))

        # Spot Price Line
        fig.add_vline(x=spot, line_dash="dash", line_color="orange", 
                      annotation_text=f"Current Spot: {spot:.2f}")

        fig.update_layout(
            xaxis_title="Stock Price ($)",
            yaxis_title="Probability Density",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Optional: Check for normalization (Area should be ~1.0)
        # Use a much higher upper bound (e.g., 10x the spot price) 
        # to ensure you capture the long right tail of the lognormal distribution
        total_area, _ = quad(mixture_pdf, 1e-6, spot * 10, args=(a1, b1, a2, b2, q))
        if abs(1.0 - total_area) > 0.05:
            st.warning(f"Note: Total distribution area is {total_area:.2f}. Model fit may be unstable.")