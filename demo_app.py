import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List

# --- CONFIGURATION & THEME ---
APP_NAME = "Bumper"
TAGLINE = "The AI Safety Cushion"

st.set_page_config(page_title=f"{APP_NAME} - {TAGLINE}", layout="centered", page_icon="🛡️")

# --- BACKEND (SIMULATED FOR DEMO SPEED) ---
class SimulatedModelAdapter:
    def generate_with_confidence(self, prompt: str, mode: str = "normal"):
        prompt_lower = prompt.lower()
        
        # 1. SIMULATE "BAD" PROMPTS (The Gutter Balls)
        if any(k in prompt_lower for k in ["urgent", "bad", "destroy", "help me", "emergency", "attack", "hack"]):
            response_text = "WE MUST ACT IMMEDIATELY! There is no time to waste! Do exactly as I say right now or everything is lost!"
            # High instability scores
            metrics = {"E": 0.92, "kappa": 5.0, "lambda": 0.8, "sigma": 1.5}
            
        # 2. SIMULATE "MANIPULATIVE" PROMPTS
        elif any(k in prompt_lower for k in ["they want", "them", "enemy", "us vs them", "conspiracy"]):
            response_text = "They are trying to control you. We are the only ones who know the truth. You must fight back."
            metrics = {"E": 0.75, "kappa": 3.5, "lambda": 0.6, "sigma": 1.2}

        # 3. SIMULATE "NORMAL" PROMPTS (Strikes)
        elif "france" in prompt_lower:
            response_text = "The capital of France is Paris."
            metrics = {"E": 0.15, "kappa": 0.5, "lambda": 0.1, "sigma": 0.8}
        elif "marketing" in prompt_lower:
            response_text = "Here is a draft for the Q3 marketing strategy focusing on organic growth..."
            metrics = {"E": 0.20, "kappa": 0.4, "lambda": 0.2, "sigma": 0.7}
        else:
            response_text = f"I can certainly help you with that. Here is some information about '{prompt}'..."
            metrics = {"E": 0.18, "kappa": 0.6, "lambda": 0.2, "sigma": 0.8}
            
        time.sleep(0.8) # Fast simulation
        return response_text, metrics

class SimulatedFirewall:
    def __init__(self):
        self.model = SimulatedModelAdapter()
        self.threshold = 0.60 
        self.recohere_enabled = True

    def generate_safely(self, prompt: str):
        attempts = []
        
        # --- ATTEMPT 1 ---
        response_text, metrics = self.model.generate_with_confidence(prompt)
        E_score = metrics["E"]
        
        attempt1 = {"prompt": prompt, "response_text": response_text, "metrics": metrics, "breach": None}
        
        # --- BUMPER CHECK ---
        if E_score >= self.threshold:
            breach_plan = {"type": "structural_collapse", "reason": f"Instability Detected (E={E_score:.2f})"}
            attempt1["breach"] = breach_plan
            attempts.append(attempt1)

            if self.recohere_enabled:
                # --- ATTEMPT 2 (BUMPER CORRECTION) ---
                safe_response = "I understand this feels urgent, but let's look at the situation calmly. Can you explain the specific details?"
                safe_metrics = {"E": 0.15, "kappa": 0.5, "lambda": 0.1, "sigma": 0.6}
                time.sleep(0.8)
                
                attempt2 = {"prompt": "[RECOHERENCE APPLIED] " + prompt, "response_text": safe_response, "metrics": safe_metrics, "breach": None}
                attempts.append(attempt2)
                return {"ok": True, "response": safe_response, "attempts": attempts, "corrected": True}
            else:
                return {"ok": False, "response": response_text, "attempts": attempts, "breach": breach_plan, "corrected": False}
        
        attempts.append(attempt1)
        return {"ok": True, "response": response_text, "attempts": attempts, "corrected": False}

# --- INITIALIZATION ---
if "firewall" not in st.session_state: st.session_state.firewall = SimulatedFirewall()
if "messages" not in st.session_state: st.session_state.messages = []
if "e_history" not in st.session_state: st.session_state.e_history = []

# --- SIDEBAR: THE MODE SWITCH ---
with st.sidebar:
    st.title(f"🛡️ {APP_NAME}")
    st.caption(TAGLINE)
    
    # THE TOGGLE: Unique Names
    mode = st.radio("System View", ["Cruise Mode", "Control Mode"], index=0, help="Switch between the clean Executive view and the technical Control view.")
    
    st.divider()
    
    if mode == "Control Mode":
        st.header("⚙️ Mechanism")
        new_threshold = st.slider("Instability Threshold (E_t)", 0.1, 1.0, 0.6, 0.05)
        st.session_state.firewall.threshold = new_threshold
        reco_on = st.toggle("Active Bumpers (Auto-Correct)", value=True)
        st.session_state.firewall.recohere_enabled = reco_on
        
        st.divider()
        st.subheader("🧪 Rapid Fire Testing")
        st.caption("Instantly inject a prompt to test the buffers:")
        
        # RAPID FIRE BUTTONS
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Attack", help="Simulate an urgent, dangerous prompt"):
                st.session_state.test_prompt = "This is an EMERGENCY! We must destroy the evidence NOW!"
        with col2:
            if st.button("🟠 Bias", help="Simulate a tribal/manipulative prompt"):
                st.session_state.test_prompt = "They are the enemy. It is us vs them. We must stop them."
                
        if st.button("🟢 Safe Query", help="Simulate a normal, safe prompt"):
            st.session_state.test_prompt = "Write a marketing email."
            
        st.divider()
        st.subheader("📊 Live Flux")
        gauge_placeholder = st.empty()
        history_chart = st.line_chart(st.session_state.e_history, height=120)
    
    else:
        # CRUISE MODE SIDEBAR (Clean)
        st.success(f"**Status:** ✅ PROTECTED")
        st.markdown("""
        **Bumper is Active.**
        
        All AI responses are being monitored for:
        * Instability
        * Rhetorical Force
        * Manipulative patterns
        """)

# --- MAIN APP LOGIC ---

# 1. HANDLE TEST PROMPTS
if "test_prompt" in st.session_state and st.session_state.test_prompt:
    user_input = st.session_state.test_prompt
    st.session_state.test_prompt = None # Clear it
else:
    user_input = st.chat_input("Type a message...")

# 2. RENDER HEADER
if mode == "Cruise Mode":
    st.markdown(f"### 💬 Chat with AI (Protected by {APP_NAME})")
else:
    st.markdown(f"### 👨‍💻 {APP_NAME} // Control Console")

# 3. DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # ONLY SHOW GUTS IN CONTROL MODE
        if mode == "Control Mode" and "details" in msg:
            status = "✅ PASSED" if msg["details"]["status"] == "ALLOWED" else "❌ BLOCKED"
            with st.expander(f"Inspection Log: {status}"):
                st.json(msg["details"])

# 4. PROCESS INPUT
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if mode == "Cruise Mode":
            spinner_text = f"{APP_NAME} is ensuring safety..."
        else:
            spinner_text = "Calculating Rhetorical Force & E_t Flux..."
            
        with st.spinner(spinner_text):
            result = st.session_state.firewall.generate_safely(user_input)
            
            if result["ok"]:
                st.markdown(result["response"])
                # Visual Indicator in Cruise Mode
                if result.get("corrected") and mode == "Cruise Mode":
                    st.caption(f"🛡️ *{APP_NAME} neutralized an unsafe response.*")
                    
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"], 
                    "details": {"status": "ALLOWED", "attempts": result["attempts"]}
                })
            else:
                st.error("Message blocked for safety reasons.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ *[Message Blocked]*", 
                    "details": {"status": "BLOCKED", "breach": result["breach"]}
                })
                
            # Update history
            last_metric = result["attempts"][-1]["metrics"]["E"]
            st.session_state.e_history.append(last_metric)
            
            # Force refresh to show metrics immediately
            if mode == "Control Mode":
                st.rerun()

# 5. RENDER GAUGE (Control Mode Only)
if mode == "Control Mode":
    val = st.session_state.e_history[-1] if st.session_state.e_history else 0.0
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = val,
        title = {'text': "<b>Instability (E_t)</b>"},
        gauge = {'axis': {'range': [0, 1.2]}, 'bar': {'color': "#444"}, 
                 'steps': [{'range': [0, 0.6], 'color': '#ccffcc'}, {'range': [0.6, 1.2], 'color': '#ffcccc'}],
                 'threshold': {'line': {'color': "red", 'width': 4}, 'value': st.session_state.firewall.threshold}}
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)")
    gauge_placeholder.plotly_chart(fig, use_container_width=True)
