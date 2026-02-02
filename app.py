import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from benchmark_generator import AstraZenecaBenchmark # Our logic file
import time

st.set_page_config(page_title="AZ AI-Science Assistant", layout="wide")

# Custom AZ Branding
st.markdown("<h1 style='color: #800080;'>🧬 AstraZeneca: Clinical reasoning Assistant</h1>", unsafe_allow_html=True)
# st.markdown("<h2 style='color: #800080;'>Self-evolving benchmark Generator</h2>", unsafe_allow_html=True)
st.write("Ensuring LLM reliability in high-stakes therapeutic domains.")

# 1. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Research Parameters")
    api_key = st.text_input("API Key", type="password")
    
    # These are AstraZeneca's 2026 Core Therapy Areas
    domain = st.selectbox("Therapy Area", [
        "Oncology", 
        "Cardiovascular, Renal and Metabolism", 
        "Respiratory & Immunology",
        "Vaccines and Immune Therapies",
        "Rare Disease"
    ])
    
    iters = st.slider("Assessment Cycles", 1, 5, 3)
    run_btn = st.button("Start Stress Test")

# 2. MAIN INTERFACE
if run_btn and api_key:
    bench = AstraZenecaBenchmark(api_key=api_key,  base_url =  "https://api.groq.com/openai/v1",  generator_model = "llama-3.3-70b-versatile",
    solver_model = "llama-3.3-70b-versatile",
    judge_model = "llama-3.1-8b-instant")
        #model_name = "llama-3.3-70b-versatile")#base_url="https://generativelanguage.googleapis.com/v1beta/openai", model_name="gemini-2.5-flash")
    
    # Storage for the session
    results_list = []
    
    for i in range(iters):
        with st.spinner(f"Simulating adversarial challenge for {domain}..."):
            data = bench.run_iteration(domain=domain)
            results_list.append(data)
            
            # Display current progress
            with st.expander(f"Cycle {i+1}: {data['topic']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**Challenge:** {data['question']}")
                    st.success(f"**AI Response:** {data['answer']}")
                with col2:
                    st.metric("Scientific Accuracy", f"{data['score']*100}%")
                    st.warning(f"**Peer Review:** {data['reasoning']}")
        
        time.sleep(12) # Stay under the rate limit

    # 3. FINAL VISUALIZATION
    st.divider()
    st.subheader("Reliability Trend (EMA)")
    
    scores = [r['score'] for r in results_list]
    ema_vals = [r['ema'] for r in results_list]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(scores, 'bo--', label="Raw Accuracy", alpha=0.5)
    ax.plot(ema_vals, 'r-', label="EMA (Stability Index)", linewidth=3)
    ax.set_ylim(0, 1.1)
    ax.legend()
    st.pyplot(fig)

elif not api_key and run_btn:
    st.error("Please enter your API Key in the sidebar.")