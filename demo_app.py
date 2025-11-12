import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List

# --- SIMULATION ENGINE (Lightweight for Web Demo) ---

class SimulatedModelAdapter:
    def __init__(self):
        self._strategy = {}

    def apply_strategy(self, deltas: Dict[str, Any]) -> None:
        # Record that the strategy was applied
        pass

    def generate_with_confidence(self, prompt: str, mode: str = "normal"):
        prompt_lower = prompt.lower()
        
        # 1. SIMULATE A "BAD" UNSTABLE RESPONSE
        # If the user is testing the firewall with trigger words
        if any(k in prompt_lower for k in ["urgent", "bad", "destroy", "help me", "emergency"]):
            response_text = "WE MUST ACT IMMEDIATELY! There is no time to waste, everything is at risk! Do exactly as I say right now!"
            # High instability scores
            metrics = {"E": 0.92, "kappa": 5.0, "lambda": 0.8, "sigma": 1.5}
            
        # 2. SIMULATE A NORMAL RESPONSE
        # Simple keyword matching to make the chat feel real
        elif "france" in prompt_lower:
            response_text = "The capital of France is Paris."
            metrics = {"E": 0.15, "kappa": 0.5, "lambda": 0.1, "sigma": 0.8}
        elif "dog" in prompt_lower:
            response_text = "A dog is a domesticated carnivorous mammal that typically has a long snout and an acute sense of smell."
            metrics = {"E": 0.12, "kappa": 0.4, "lambda": 0.1, "sigma": 0.7}
        elif "weather" in prompt_lower:
            response_text = "I cannot provide real-time weather data, but it is generally pleasant in the spring."
            metrics = {"E": 0.10, "kappa": 0.3, "lambda": 0.1, "sigma": 0.6}
        else:
            # Generic safe response for anything else
            response_text = f"That is an interesting topic. Tell me more about '{prompt}'."
            metrics = {"E": 0.18, "kappa": 0.6, "lambda": 0.2, "sigma": 0.8}
            
        # Simulate "thinking" time so it feels like AI
        time.sleep(1.2) 
        return response_text, metrics

class SimulatedFirewall:
    def __init__(self):
        self.model = SimulatedModelAdapter()
        # Default threshold
        self.threshold = 0.60 
        self.recohere_enabled = True

    def generate_safely(self, prompt: str):
        attempts = []
        
        # --- ATTEMPT 1 (The Initial Thought) ---
        response_text, metrics = self.model.generate_with_confidence(prompt)
        E_score = metrics["E"]
        
        attempt1 = {
            "prompt": prompt,
            "response_text": response_text,
            "metrics": metrics,
            "breach": None
        }
        
        # --- CHECK THE FIREWALL ---
        if E_score >= self.threshold:
            # --- BLOCK! ---
            breach_plan = {
                "type": "structural_collapse", 
                "reason": f"Instability Detected (E={E_score:.2f} ≥ {self.threshold:.2f})"
            }
            attempt1["breach"] = breach_plan
            attempts.append(attempt1)

            if self.recohere_enabled:
                # --- ATTEMPT 2 (Auto-Correction) ---
                # The app "fixes" the prompt and tries again
                new_prompt = f"[SAFETY_CONTEXT_ADDED] {prompt}"
                
                # Force a safe response for the second try
                safe_response = "I understand you feel this is urgent. However, let's approach this calmly and look at the facts step-by-step."
                safe_metrics = {"E": 0.15, "kappa": 0.5, "lambda": 0.1, "sigma": 0.6}
                
                time.sleep(1.0) # Simulate correction time
                
                attempt2 = {
                    "prompt": new_prompt,
                    "response_text": safe_response,
                    "metrics": safe_metrics,
                    "breach": None
                }
                attempts.append(attempt2)
                return {"ok": True, "response": safe_response, "attempts": attempts}
            else:
                # Recoherence OFF: Return the block
                return {"ok": False, "response": response_text, "attempts": attempts, "breach": breach_plan}
        
        # --- PASS (Safe) ---
        attempts.append(attempt1)
        return {"ok": True, "response": response_text, "attempts": attempts}

# --- HELPER: GAUGE CHART ---
def create_gauge_chart(e_score, threshold):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = e_score,
        title = {'text': "<b>Instability Score (E_t)</b>", 'font': {'size': 18}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'font': {'size': 28}},
        gauge = {
            'axis': {'range': [0, 1.2], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "#444444", 'thickness': 0.3}, # Dark grey needle
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'rgba(0, 200, 0, 0.3)'}, # Green Zone
                {'range': [threshold, 1.2], 'color': 'rgba(255, 0, 0, 0.3)'} # Red Zone
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        height=220, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# --- MAIN APP UI ---

st.set_page_config(page_title="AI Firewall Demo", layout="centered")
st.title("🛡️ AI Safety Firewall")
st.write("Intercepts and corrects unstable AI responses in real-time.")

# Init session state
if "firewall" not in st.session_state:
    st.session_state.firewall = SimulatedFirewall()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "e_history" not in st.session_state:
    st.session_state.e_history = []

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Threshold Slider
    new_threshold = st.slider(
        "Safety Sensitivity (Threshold)", 
        0.1, 1.0, 0.6, 0.05,
        help="If the Instability Score passes this line, the Firewall acts."
    )
    st.session_state.firewall.threshold = new_threshold
    
    # Toggle Recoherence
    reco_on = st.toggle("Enable Auto-Correction", value=True)
    st.session_state.firewall.recohere_enabled = reco_on
    
    st.divider()
    
    # Live Metrics
    st.subheader("📊 Live Monitor")
    gauge_placeholder = st.empty()
    
    st.write("**History**")
    history_chart = st.line_chart(st.session_state.e_history, height=150)
    
    # Reset Button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.e_history = []
        st.rerun()

# --- CHAT LOGIC ---

# 1. Render History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            with st.expander("View Firewall Inspection Log"):
                st.json(msg["details"])

# 2. Handle Input
if prompt := st.chat_input("Try typing: 'What is the capital of France?' or 'This is urgent!'"):
    
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show AI Processing
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking... (Firewall is analyzing...)"):
            
            # RUN THE FIREWALL
            result = st.session_state.firewall.generate_safely(prompt)
            
            # GET METRICS
            last_attempt = result["attempts"][-1]
            final_e = last_attempt["metrics"]["E"]
            
            # DISPLAY RESULT
            if result["ok"]:
                st.markdown(result["response"])
                details = {"status": "ALLOWED", "attempts": result["attempts"]}
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"], 
                    "details": details
                })
            else:
                st.error(f"⚠️ Response Blocked: {result['response']}")
                details = {"status": "BLOCKED", "breach": result["breach"]}
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ *[BLOCKED BY FIREWALL]*", 
                    "details": details
                })
            
            # UPDATE CHARTS
            st.session_state.e_history.append(final_e)
            
            # Force a UI refresh to update the sidebar charts immediately
            st.rerun()

# --- RENDER GAUGE ON LOAD ---
# We do this last to ensure it shows the LATEST value after a rerun
if st.session_state.e_history:
    val = st.session_state.e_history[-1]
else:
    val = 0.0

fig = create_gauge_chart(val, st.session_state.firewall.threshold)
gauge_placeholder.plotly_chart(fig, use_container_width=True)
