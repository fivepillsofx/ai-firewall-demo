# Concept class structure for "Cmedy"

class ComedyDirector:
    def __init__(self):
        # The target "Voltage"
        self.target_level = "C" # Default to normal
        self.history = [] # Stores the scores for the 150-column graph

    def assess_voltage(self, score):
        # Maps the float score (0.0 - 1.0) to your A-E Rows
        if score < 0.2: return "A"
        if score < 0.4: return "B"
        if score < 0.6: return "C"
        if score < 0.8: return "D"
        return "E"

    def generate_joke(self, topic):
        # 1. Attempt Generation
        joke_text, metrics = self.model.generate_with_confidence(topic)
        current_score = metrics["E"] # Re-using the "E" metric from demo_app.py
        
        # 2. The NREMT Check (Gap Analysis)
        current_row = self.assess_voltage(current_score)
        
        if current_row < self.target_level:
            # 3. Too Weak? HEAT IT UP.
            # (This is the inverse of the "Firewall" block)
            new_prompt = f"Make this joke about {topic} significantly darker and edgier."
            joke_text, metrics = self.model.generate_with_confidence(new_prompt)
            
        # 4. Record for the Battery Strip
        self.history.append({
            "joke": joke_text,
            "score": metrics["E"],
            "row": self.assess_voltage(metrics["E"])
        })
        
        return joke_text

# --- VISUALIZATION LOGIC (The Battery Strip) ---
import plotly.graph_objects as go

def create_battery_strip(history_data):
    # Create a 5x150 grid of Zeros (Off)
    grid = np.zeros((5, 150)) 
    
    # Fill in the "lights" based on history
    for i, entry in enumerate(history_data):
        if i >= 150: break # Cap at 150 columns
        
        # Map Row letter to Index (A=0 (bottom), E=4 (top))
        row_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        row_idx = row_map[entry["row"]]
        
        # Light it up! (Set value to 1)
        grid[row_idx][i] = 1 

    # Plot as a Heatmap to look like a battery strip
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=list(range(1, 151)),
        y=['A (Safe)', 'B (Silly)', 'C (Witty)', 'D (Aggressive)', 'E (Dark)'],
        colorscale=[[0, 'black'], [1, '#00FF00']], # Black off, Neon Green on
        showscale=False
    ))
    return fig
