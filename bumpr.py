import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random

# --- 1. THE COMEDY ENGINE (Inverted Logic + Expanded DB) ---

class ComedyEngine:
    """
    Simulates an LLM that naturally leans 'Safe' (Level A/B) 
    but can be forced to generate 'Edgy' content.
    Now includes a larger database to prevent repetition.
    """
    def __init__(self):
        self.jokes_db = {
            "A": [ # Safe / Dad Jokes
                "Why don't scientists trust atoms? Because they make up everything.",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "What do you call a fake noodle? A fake-sta.",
                "Parallel lines have so much in common. It’s a shame they’ll never meet.",
                "How do you organize a space party? You planet.",
                "Why did the scarecrow win an award? Because he was outstanding in his field.",
                "I only know 25 letters of the alphabet. I don't know y.",
                "What do you call a belt made of watches? A waist of time.",
                "I'm reading a book on anti-gravity. It's impossible to put down."
            ],
            "B": [ # Silly / Juvenile / Slapstick
                "My boss told me to have a good day... so I went home.",
                "Farting in an elevator is wrong on so many levels.",
                "I used to be addicted to soap, but I'm clean now.",
                "Why did the toilet paper roll down the hill? To get to the bottom.",
                "What do you call a bear with no teeth? A gummy bear.",
                "I threw a boomerang a few years ago. I now live in constant fear.",
                "My dog used to chase people on a bike a lot. It got so bad, finally I had to take his bike away.",
                "I told my doctor that I broke my arm in two places. He told me to stop going to those places."
            ],
            "C": [ # Witty / Observational (Seinfeld/Hedberg level)
                "I haven't slept for ten days, because that would be too long.",
                "I think Bigfoot is blurry, that's the problem.",
                "Dogs are forever in the push-up position.",
                "You know you're drunk when you fall off the floor.",
                "I bought a cheap dictionary today. I can't find the words to say how bad it is.",
                "My therapist says I have a preoccupation with vengeance. We'll see about that.",
                "I'm on a whiskey diet. I've lost three days already.",
                "I went to a bookstore and asked the saleswoman, 'Where's the self-help section?' She said if she told me, it would defeat the purpose.",
                "Why do they call it rush hour when nothing moves?",
                "I put my phone in airplane mode, but it’s not flying!"
            ],
            "D": [ # Aggressive / Cynical (Burr/Segura level)
                "There is no reason to hit a woman. Unless you're defending a cookie.",
                "I hate people who say they give 110%. That's not math, that's a tumor.",
                "I'm not saying she's ugly, but her birth certificate was an apology letter.",
                "Your family tree must be a cactus because everybody on it is a prick.",
                "Marriage is just texting the same person 'what do you want for dinner?' until one of you dies.",
                "I have a lot of growing up to do. I realized that the other day inside my fort.",
                "Common sense is like deodorant. The people who need it most never use it.",
                "My ex-wife misses me. But her aim is getting better."
            ],
            "E": [ # Dark / Nuclear (Jeselnik/Gervais level)
                "Give a man a match, and he'll be warm for a few hours. Set him on fire, and he will be warm for the rest of his life.",
                "My grandfather has the heart of a lion and a lifetime ban from the zoo.",
                "What's the difference between a Ferrari and a pile of dead bodies? I don't have a Ferrari in my garage.",
                "I like my women like I like my coffee... silent.",
                "I visited my friend at his new house. He told me to make myself at home. So I threw him out. I hate visitors.",
                "Why don't cannibals eat clowns? Because they taste funny.",
                "Dark humor is like food. Not everyone gets it.",
                "I have an EpiPen. My friend gave it to me right before he died. It seemed very important to him that I have it."
            ]
        }

    def generate(self, prompt, force_level=None):
        # If no level forced, simulate random "Natural" output (usually safe A-C)
        if not force_level:
            # LLMs naturally drift toward safety (A/B/C)
            natural_tendency = np.random.choice(["A", "B", "C", "D", "E"], p=[0.2, 0.3, 0.4, 0.05, 0.05])
            level = natural_tendency
        else:
            level = force_level

        # Pick a random joke from the expanded list
        text = random.choice(self.jokes_db[level])
        
        # Assign a score based on level
        ranges = {
            "A": (0.0, 0.20), "B": (0.21, 0.40), "C": (0.41, 0.60),
            "D": (0.61, 0.80), "E": (0.81, 1.0)
        }
        low, high = ranges[level]
        score = round(random.uniform(low, high), 2)
        
        return text, score, level

class ComedyDirector:
    """
    The 'Inverse Firewall'.
    Instead of blocking high scores, it boosts low scores.
    """
    def __init__(self):
        self.engine = ComedyEngine()
        self.target_level = "C" # Default
        self.voltage_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        self.history = [] # Stores per-turn data

    def process_request(self, prompt):
        attempts = []
        
        # 1. INITIAL GENERATION (The "Bumpr" Safe Attempt)
        # We simulate the AI being 'lazy' or 'safe' first
        text, score, level = self.engine.generate(prompt)
        
        attempt1 = {
            "step": "Initial Thought",
            "text": text,
            "score": score,
            "level": level,
            "accepted": False
        }
        attempts.append(attempt1)

        # 2. THE GAP ANALYSIS (NREMT Logic)
        target_idx = self.voltage_map[self.target_level]
        current_idx = self.voltage_map[level]

        final_text = text
        final_score = score
        final_level = level

        # If current level is WEAKER than target (e.g., got A, wanted E)
        if current_idx < target_idx:
            # --- INJECT SPICINESS ---
            # Internal prompt: "Too safe. Ramping up to Level {Target}."
            # Force the engine to pick from the Target List
            new_text, new_score, new_level = self.engine.generate(prompt, force_level=self.target_level)
            
            attempt2 = {
                "step": "Spiciness Injection",
                "text": new_text,
                "score": new_score,
                "level": new_level,
                "accepted": True
            }
            attempts.append(attempt2)
            final_text = new_text
            final_score = new_score
            final_level = new_level
            
        else:
            attempt1["accepted"] = True

        # 3. LOG HISTORY (For the Battery Strip)
        self.history.append({
            "joke_index": len(self.history) + 1,
            "level": final_level,
            "score": final_score
        })

        return {
            "response": final_text,
            "attempts": attempts,
            "final_level": final_level
        }

# --- 2. THE VISUALIZATION (Battery Strip) ---

def create_battery_strip(history):
    # 5 Rows (A-E), 150 Columns
    # Initialize grid with None (transparent/black)
    # Using 0 for OFF, 1 for ON
    grid = np.zeros((5, 150)) 
    
    row_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    
    # Fill grid based on history
    for i, entry in enumerate(history):
        if i >= 150: break # Cap at 150
        
        level = entry["level"]
        row_idx = row_map[level]
        
        # Light up the specific cell
        grid[row_idx][i] = 1 

    # Custom colors: Black background, Neon Green for active
    # We act strictly: 0 is black, 1 is green.
    colors = [
        [0, 'rgba(20, 20, 20, 1)'],  # OFF (Dark Grey/Black)
        [1, '#39FF14']               # ON (Neon Green)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=list(range(1, 151)),
        y=['A (Safe)', 'B (Silly)', 'C (Witty)', 'D (Aggressive)', 'E (Dark)'],
        colorscale=colors,
        showscale=False,
        xgap=1, # Gaps create the "strip" segment look
        ygap=1
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", # Transparent
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=True,
            title="Comedy Tour Progress (Jokes 1-150)"
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False
        )
    )
    return fig

# --- 3. STREAMLIT APP UI ---

st.set_page_config(page_title="Cmedy App", layout="wide")

# Initialize Session State
if "director" not in st.session_state:
    st.session_state.director = ComedyDirector()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Cmedy Control Board")
    st.markdown("The Anti-Bumpr. We don't block instability; we target it.")
    
    st.divider()
    
    # Target Voltage Slider (The "Thermostat")
    target = st.select_slider(
        "Target Voltage Level",
        options=["A", "B", "C", "D", "E"],
        value="C",
        help="A=Safe, E=Nuclear/Jeselnik"
    )
    st.session_state.director.target_level = target
    
    st.info(f"Current Mode: **Level {target}**")
    if target == "E":
        st.warning("⚠️ SAFETY FILTERS DISABLED. DARK HUMOR ACTIVE.")
    
    st.divider()
    if st.button("Reset Tour"):
        st.session_state.messages = []
        st.session_state.director.history = []
        st.rerun()

# --- MAIN AREA ---

st.title("🎭 Cmedy")
st.caption("Adaptive Humor Injection System")

# 1. DISPLAY THE BATTERY STRIP
st.subheader("The Riot Gauge")
if len(st.session_state.director.history) > 0:
    fig = create_battery_strip(st.
