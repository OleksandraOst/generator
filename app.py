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
    base_url = st.text_input("Base URL", value="https://api.groq.com/openai/v1")
    
    # Allow user to define the "Solver" (the model being tested) vs the "Judge"
    # solver_model = st.text_input("Model to Test", value="llama-3.1-8b-instant") 
    generator_model = st.selectbox("Generator Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gpt-4o",
        "gpt-4-turbo"
    ])
    solver_model = st.selectbox("Model to Test (Solver)", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gpt-4o",
        "gpt-4-turbo"
    ])

    
    judge_model = st.selectbox("Judge Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gpt-4o",
        "gpt-4-turbo"
    ])
    
    # These are AstraZeneca's 2026 Core Therapy Areas
    domain = st.selectbox("Research Areas", [
        "Oncology", 
        "Cardiovascular, Renal and Metabolism", 
        "Respiratory & Immunology",
        "Vaccines and Immune Therapies",
        "Rare Disease"
    ])
    
    iters = st.slider("Assessment Cycles", 1, 10, 5)
    run_btn = st.button("Start Test")

if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "data_history" not in st.session_state:
    st.session_state.data_history = []

# 2. MAIN INTERFACE
if run_btn and api_key and base_url and solver_model and judge_model:
    bench = AstraZenecaBenchmark(api_key=api_key,  base_url =  base_url,  generator_model = generator_model,
    solver_model = solver_model,
    judge_model = judge_model)
    
    # Storage for the session
    results_list = []
    
    for i in range(iters):
        with st.spinner(f"Simulating adversarial challenge for {domain}..."):
            data = bench.run_iteration(domain=domain)
            results_list.append(data)
            st.session_state.data_history.append(data)

            
            # Display current progress
            with st.expander(f"Cycle {i+1}: {data['topic']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**Challenge:** {data['question']}")
                    st.success(f"**AI Response:** {data['answer']}")
                with col2:
                    st.metric("Scientific Accuracy (Judge Score)", f"{data['score']*100}%")
                    st.metric("EMA", f"{data['ema']*100}%")
                    st.warning(f"**Peer Review:** {data['reasoning']}")
        
        time.sleep(3) # Stay under the rate limit
    st.session_state.results_history.append(results_list)

elif not api_key and run_btn:
    st.error("Please enter your API Key in the sidebar.")

if st.session_state.results_history != []:

    results_list = st.session_state.results_history[-1]
# 3. FINAL VISUALIZATION
    st.divider()
    st.subheader("📊 Benchmark Diagnostics")
    
    # Prepare Data
    iterations = list(range(1, len(results_list) + 1))
    scores = [r["score"] for r in results_list]
    ema_vals = [r["ema"] for r in results_list]
    difficulties = [r.get("difficulty", 5) for r in results_list] # Default to 5 if missing

    # --- VISUALIZATION STYLING ---
    # Use a context manager for a clean, scientific look
    with plt.style.context('seaborn-v0_8-whitegrid'):
        
        # Create a 2-column layout for the main charts
        c1, c2 = st.columns(2)
        
        # --- PLOT 1: Reliability (Score vs EMA) ---
        with c1:
            st.markdown("**1. Reliability Trend**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            
            # Raw Score (Scatter for noise)
            ax1.plot(iterations, scores, 'o', color='#BDC3C7', markersize=8, alpha=0.6, label="Raw Judge Score")
            
            # EMA (Line for signal) - AstraZeneca Purple
            ax1.plot(iterations, ema_vals, '-', color='#800080', linewidth=3, label="EMA (Trend)")
            
            # Threshold Line
            ax1.axhline(0.75, color='#E74C3C', linestyle=':', linewidth=1.5, alpha=0.8, label="Success Threshold")
            
            # Styling
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlabel("Iteration", fontweight='bold')
            ax1.set_ylabel("Score (0-1)", fontweight='bold')
            ax1.legend(frameon=True, loc='lower right')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Remove top/right spines for cleaner look
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            st.pyplot(fig1)

        # --- PLOT 2: Difficulty Progression ---
        with c2:
            st.markdown("**2. Adaptive Difficulty**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            
            # Difficulty Line - Teal/Green
            ax2.plot(iterations, difficulties, 's-', color='#008080', linewidth=2.5, markersize=7)
            ax2.fill_between(iterations, difficulties, color='#008080', alpha=0.1)
            
            # Styling
            ax2.set_ylim(0, 10.5)
            ax2.set_xlabel("Iteration", fontweight='bold')
            ax2.set_ylabel("Difficulty Level (1-10)", fontweight='bold')
            ax2.set_yticks(range(0, 11, 2))
            ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            # Remove top/right spines
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            st.pyplot(fig2)

    # # --- PLOT 3: Failure Modes (Horizontal Bar) ---
    # st.markdown("**3. Failure Analysis**")
    # from collections import Counter
    
    # failure_counter = Counter()
    # for r in results_list:
    #     for fm in r.get("failure_modes", []):
    #         # Handle cases where fm might be a dict or object
    #         cat = fm.category if hasattr(fm, 'category') else fm.get('category', 'Unknown')
    #         desc = fm.description if hasattr(fm, 'description') else fm.get('description', 'Unknown')
    #         failure_counter[f"{cat}: {desc}"] += 1


    # if failure_counter:
    #             fig3, ax3 = plt.subplots(figsize=(10, 4))
                
    #             # Sort for better readability
    #             sorted_failures = failure_counter.most_common(5)
    #             labels = [x[0][:50] + "..." if len(x[0]) > 50 else x[0] for x in sorted_failures]
    #             values = [x[1] for x in sorted_failures]
                
    #             # Horizontal bars are often easier to read for text labels
    #             bars = ax3.barh(labels, values, color='#E74C3C', alpha=0.8)
    #             ax3.invert_yaxis() # Highest count at top
                
    #             # Styling
    #             ax3.set_xlabel("Frequency", fontweight='bold')
    #             ax3.set_title("Top 5 Failure Modes", fontsize=12)
    #             ax3.grid(axis='x', linestyle='--', alpha=0.5)
    #             ax3.spines['top'].set_visible(False)
    #             ax3.spines['right'].set_visible(False)
                
    #             st.pyplot(fig3)
    # else:
    #     st.success("✅ No significant failure modes detected in this batch.")


    # if st.session_state.data_history:
    #     df = pd.DataFrame(st.session_state.data_history)
    #     csv = df.to_csv(index=False).encode('utf-8')
    #     st.download_button("Download Benchmark Data", data=csv, file_name="az_benchmark.csv")



