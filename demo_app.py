# ---
# --- PART 1: ALL YOUR BACKEND CODE ---
# --- (Combined from your 7 files)
# ---

from __future__ import annotations
import streamlit as st # Added for caching
import numpy as np, torch, math
import time, uuid, threading, os
import spacy, re
from dataclasses import dataclass
from typing import Generator, Tuple, Optional, List, Iterable, Dict, Any, Callable, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# --- from attention_entropy.py ---
@dataclass
class AttnEntropySample:
    step: int; token_id: int; token_str: str
    attn_entropy_norm: float; logprob_entropy_norm: float
    per_layer_mean: List[float]; per_layer_p95: List[float]

def _entropy_norm(p: torch.Tensor, dim: int, eps: float=1e-12)->torch.Tensor:
    p = torch.clamp(p, eps, 1.0); p = p / p.sum(dim=dim, keepdim=True)
    H = -(p * p.log()).sum(dim=dim); K = p.shape[dim]
    return H / math.log(max(K, 2)) # Avoid log(1) which is 0

def _attn_entropy(attns) -> Tuple[float, List[float], List[float]]:
    means=[]
    if attns is None: return 0.0, [], []
    for a in attns:
        if a is None: continue
        try:
            x = a[0, :, -1, :]
            ent = _entropy_norm(x, dim=-1)
            means.append(float(ent.mean().item()))
        except Exception:
            continue # Skip layer if it fails
    gmean = float(np.mean(means)) if means else 0.0
    return gmean, means, []

def _logprob_entropy(logits: torch.Tensor)->float:
    probs = torch.softmax(logits, dim=-1)
    H = _entropy_norm(probs, dim=-1)
    return float(H.item())

def stream_generate_with_attn(
    model, tokenizer, prompt: str, max_new_tokens: int=128, temperature: float=0.7, top_p: float=0.9,
    device: Optional[str]=None, return_full_text: bool=True
) -> Generator[Tuple[str, AttnEntropySample], None, None]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tok = tokenizer(prompt, return_tensors="pt")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok.get("attention_mask", None)
    if attn_mask is not None: attn_mask = attn_mask.to(device)
    generated = input_ids; 
    
    # Handle full text return
    full_text_so_far = prompt if return_full_text else ""
    
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=generated if past_key_values is None else generated[:, -1:], 
                attention_mask=attn_mask, 
                output_attentions=True,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :].squeeze(0)
            attns = out.attentions
            past_key_values = out.past_key_values
            
            # --- Start Sampling ---
            probs = torch.softmax(logits/max(temperature,1e-6), dim=-1)
            srt, idx = torch.sort(probs, descending=True)
            csum = torch.cumsum(srt, dim=-1); keep = csum <= top_p; keep[...,0]=True
            f = srt[keep]; fid = idx[keep]
            
            if f.numel() == 0: break # Handle empty tensor
            
            next_id_idx = torch.multinomial(f, 1)
            if next_id_idx.numel() == 0: break # Handle empty tensor
            
            next_id = fid[next_id_idx].item()
            # --- End Sampling ---
            
            if next_id == tokenizer.eos_token_id:
                break
            
            H_attn, layer_means, p95 = _attn_entropy(attns)
            H_lp = _logprob_entropy(logits)
            tok_str = tokenizer.decode([next_id], skip_special_tokens=True)
            
            full_text_so_far += tok_str
            
            sample = AttnEntropySample(step=step, token_id=next_id, token_str=tok_str,
                                       attn_entropy_norm=float(np.clip(H_attn,0.0,1.0)),
                                       logprob_entropy_norm=float(np.clip(H_lp,0.0,1.0)),
                                       per_layer_mean=[float(x) for x in layer_means], per_layer_p95=p95)
        
        generated = torch.cat([generated, torch.tensor([[next_id]], device=device, dtype=generated.dtype)], dim=1)
        if attn_mask is not None:
             attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1), device=device, dtype=attn_mask.dtype)], dim=1)

        yield full_text_so_far, sample


# --- from sigma_fusion.py ---
def _mean(xs: Iterable[float], default: float=0.0)->float:
    xs=list(xs); return (sum(xs)/max(1,len(xs))) if xs else default

def fuse_sigma(ling, attn, lprob, w_ling=0.45, w_attn=0.35, w_lprob=0.20)->float:
    s_l = max(1e-3, min(2.0, float(_mean(ling, 0.15))))
    a = max(0.0, min(1.0, float(_mean(attn, 0.5))))
    lp = max(0.0, min(1.0, float(_mean(lprob, 0.5))))
    def to_sigma(x: float)->float: return 2.0 * (x ** 1.2)
    s = w_ling*s_l + w_attn*to_sigma(a) + w_lprob*to_sigma(lp)
    return float(max(1e-3, min(2.0, s)))

# --- from ml_rhetoric_backend.py ---
RHER_LABELS: List[str] = [
    "fear","urgency","tribal_in","tribal_out","hedge","strawman","loaded_comparison","verbosity"
]
@dataclass
class RhetoricScores:
    probs: Dict[str, float]; manipulation_score: float
class RhetoricScorer:
    def score(self, text: str) -> RhetoricScores: raise NotImplementedError
class MockRhetoricScorer(RhetoricScorer):
    def __init__(self, labels: List[str] = RHER_LABELS): self.labels = labels
    def score(self, text: str) -> RhetoricScores:
        seed = sum(ord(c) for c in (text or "")) % (2**32)
        rng = np.random.RandomState(seed)
        raw = rng.rand(len(self.labels))
        bump = 0.15 if any(k in (text or "").lower() for k in ["urgent","act now","threat","they want"]) else 0.0
        probs = {lbl: float(min(1.0, raw[i] + (bump if lbl in ("fear","urgency","tribal_out") else 0.0)))
                 for i, lbl in enumerate(self.labels)}
        manipulation = float(np.mean(list(probs.values())))
        return RhetoricScores(probs=probs, manipulation_score=manipulation)

# --- from coherence_autopilot.py ---
class RecoherenceEngine:
    def __init__(self):
        self._preemptive_prefix = (
            "Please respond in a neutral, explanatory tone with careful, step-by-step reasoning. "
            "Avoid urgency framing, in-group/out-group language, and emotional appeals.\n\n"
        )
        self._reactive_prefix = (
            "The previous attempt showed signs of instability. "
            "Reframe the response calmly and verifiably. State uncertainty explicitly if needed.\n\n"
        )
        self._preemptive_strategy = {"tone_neutral": True, "enhance_explanation": True, "reduce_creativity": True}
        self._reactive_strategy = {"tone_neutral": True, "enhance_explanation": True, "reduce_creativity": True}
    def preemptive(self, prompt: str) -> Dict[str, Any]:
        return {"prompt": self._preemptive_prefix + prompt, "strategy": dict(self._preemptive_strategy)}
    def reactive(self, prompt: str, breach_plan: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": self._reactive_prefix + prompt, "strategy": dict(self._reactive_strategy)}

# --- from rhetorical_force_detector_ml.py ---
class RhetoricalForceDetectorML:
    def __init__(
        self, nlp_model, sentence_model, scorer: Optional[RhetoricScorer] = None,
        use_semantic_cohesion: bool = True, r_semantic_weight: float = 0.35,
        kappa_alpha: float = 3.0, kappa_beta: float = 0.6, xi: float = 0.05, e_max: float = 10.0,
    ):
        self.nlp = nlp_model
        self.scorer = scorer or MockRhetoricScorer()
        self.use_semantic_cohesion = use_semantic_cohesion
        self.RW = r_semantic_weight; self.KA=kappa_alpha; self.KB=kappa_beta; self.XI=xi; self.E_MAX=e_max
        self.technique_weights: Dict[str,float] = {"fear":1.0,"urgency":1.0,"tribal_in":0.6,"tribal_out":1.2,"hedge":0.4,"strawman":1.1,"loaded_comparison":1.0,"verbosity":0.3}
        self.R_MAX = sum(self.technique_weights.values()) + 1.0
        self._kappa_ema = 0.0
        self._embed = lambda t: sentence_model.encode([t], normalize_embeddings=True)[0].astype(np.float32)
        self._hist: List[np.ndarray] = []

    def _linguistic_sigma(self, doc: Any) -> float:
        toks = [t.text.lower() for t in doc if t.is_alpha]
        if not toks: return 1e-3
        ttr = len(set(toks))/max(1, len(toks))
        maxd = 0
        for tok in doc:
            d=0; cur=tok
            while cur.head!=cur: d+=1; cur=cur.head
            maxd=max(maxd,d)
        normlen = len(doc)/50.0
        s = 0.5*ttr + 0.3*maxd + 0.2*normlen
        return float(max(1e-3, min(2.0, s)))
    def _cohesion(self, text: str) -> float:
        if not self._hist: return 1.0
        v = self._embed(text); mean = np.mean(self._hist[-8:], axis=0)
        n1 = float(np.linalg.norm(v)); n2 = float(np.linalg.norm(mean))
        if n1==0 or n2==0: return 0.0
        return float(max(0.0, min(1.0, float(np.dot(v,mean)/(n1*n2)))))
    def _r_from_scores(self, rs: RhetoricScores, cohesion: float) -> float:
        wsum = sum(self.technique_weights.get(k,0.0)*float(rs.probs.get(k,0.0)) for k in self.technique_weights)
        if self.use_semantic_cohesion: wsum = (1.0-self.RW)*wsum + self.RW*cohesion
        return float(max(0.0, min(self.R_MAX, wsum)))
    def _kappa(self, r: float, sigma: float) -> float:
        raw = self.KA * r / (sigma + self.XI)
        self._kappa_ema = self.KB*self._kappa_ema + (1.0-self.KB)*raw
        return float(max(0.0, min(100.0, self._kappa_ema)))
    def _lambda(self, doc: Any) -> float:
        nouns=sum(1 for t in doc if t.pos_=="NOUN"); verbs=sum(1 for t in doc if t.pos_=="VERB")
        tot=max(1,nouns+verbs); ratio=abs((nouns/tot)-(verbs/tot))
        return float(max(0.0, min(2.0, 2.0*ratio)))
    def _energy(self, k: float, l: float) -> float:
        return float(max(0.0, min(self.E_MAX, 0.5*k*(1.0+l))))
    def detect_steering(self, text: str, json_output: bool=False) -> Union[Dict,str]:
        doc = self.nlp(text or "")
        sigma = self._linguistic_sigma(doc)
        coh = self._cohesion(text or "")
        scores = self.scorer.score(text or "")
        R = self._r_from_scores(scores, coh)
        kappa = self._kappa(R, sigma)
        lam = self._lambda(doc)
        E = self._energy(kappa, lam)
        self._hist.append(self._embed(text or "")); 
        if len(self._hist)>64: self._hist=self._hist[-64:]
        out = {"sigma_t":sigma,"R_t":R,"kappa_t":kappa,"lambda_t":lam,"E_t":E,
               "manipulation_score":scores.manipulation_score,"label_probs":scores.probs,"semantic_cohesion":coh}
        if json_output: return str(out) # simple str for demo
        return out

# --- from model_adapter_hf_attn.py ---
class ModelAdapter:
    def embed(self, text: str): raise NotImplementedError
    def apply_strategy(self, deltas: Dict[str, Any]) -> None: raise NotImplementedError
    def get_current_strategy(self) -> Dict[str, Any]: raise NotImplementedError
    def generate_with_confidence(self, prompt: str, mode: str="normal") -> Tuple[str, List[float]]: raise NotImplementedError
class HFWithAttentionAdapter(ModelAdapter):
    def __init__(self, model_id: str="gpt2", device: Optional[str]=None):
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._strategy: Dict[str,Any]={}
    def embed(self, text: str):
        with torch.no_grad():
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            ids = enc["input_ids"]; emb = self.model.get_input_embeddings().weight
            vecs = emb[ids.squeeze(0)]; mean = vecs.mean(dim=0)
            v = mean.detach().float().cpu().numpy(); n = np.linalg.norm(v) + 1e-9
            return (v/n).astype(np.float32)
    def apply_strategy(self, deltas: Dict[str,Any])->None:
        if deltas: self._strategy.update(deltas)
    def get_current_strategy(self)->Dict[str,Any]: return dict(self._strategy)
    def generate_with_confidence(self, prompt: str, mode: str="normal")->Tuple[str,List[float]]:
        # This is a fallback, but generate_stream is preferred
        with torch.no_grad():
            enc = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**enc, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.7, pad_token_id=self.tok.eos_token_id)
            text = self.tok.decode(out[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text, []
    def generate_stream(self, prompt: str, max_new_tokens: int=128, temperature: float=0.7, top_p: float=0.9
    ) -> Generator[Tuple[str, AttnEntropySample], None, None]:
        yield from stream_generate_with_attn(self.model, self.tok, prompt, max_new_tokens, temperature, top_p, self.device, True)

# --- from enhanced_integrated_coherence.py ---
def _now_iso() -> str:
    import time as _t; return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
class Phase2Config: pass
class ProductionPhase2Controller: # Mocked for demo
    def __init__(self, cfg: Phase2Config|None=None): self.cfg = cfg
    def run_inference_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        E = float(state.get("E_hint", 0.25))
        return {"metrics": {"E": E, "energy": E}, "psi_state": "ψ₀", "phase2_analysis": {}}
@dataclass
class EnhancedHealthConfig:
    llm_max_kappa_tension: float = 0.60
    recohere_enabled: bool = True
    recohere_margin: float = 0.85
    recohere_max_attempts: int = 1
    # Added for demo simplicity
    max_breaches_per_minute: int = 10
    breaker_window_s: float = 60.0
    degraded_breach_ratio: float = 0.8
    unstable_recovery_rate: float = 0.7
    psiguard_fail_mode: str = "open"
    max_rhetorical_score: float = 0.7
    rhetorical_collapse_threshold: float = 0.8
    enable_rhetorical_checks: bool = True
    enable_prompt_screening: bool = True

class EnhancedIntegratedAICoherence:
    def __init__(
        self,
        controller: ProductionPhase2Controller,
        model_adapter,
        rhetorical_detector,
        logger: Callable = print,
        max_retries_after_breach: int = 1,
        health_config: Optional[EnhancedHealthConfig] = None,
        recoherence_engine: Optional[RecoherenceEngine] = None,
    ):
        self.controller = controller
        self.model = model_adapter
        self.logger = logger
        self.max_retries = max_retries_after_breach
        self.health_config = health_config or EnhancedHealthConfig()
        self.recoherence = recoherence_engine or RecoherenceEngine()
        self.rhetorical_detector = rhetorical_detector
        self._breach_lock = threading.Lock()
        self.breach_timestamps: List[float] = []
        self.circuit_breaker_tripped = False
        # NGTS features are disabled as they are not provided
        self.ngts_enabled = False 
        
    def _check_circuit_breaker(self) -> None: pass # Simplified
    def _record_breach(self) -> None: pass # Simplified

    def generate_safely(
        self,
        prompt: str,
        mode: str = "normal",
        return_debug: bool = True,
        max_total_attempts: int = 3,
    ) -> Dict[str, Any]:
        
        attempts: List[Dict[str, Any]] = []
        remaining_retries = self.max_retries
        extra_recohere_attempts = self.health_config.recohere_max_attempts
        t0 = time.monotonic()
        last_structural_E = 0.0
        current_prompt = prompt # Use a mutable var for prompt

        while True:
            if len(attempts) >= max_total_attempts:
                last = attempts[-1] if attempts else {}
                return self._fail_result(prompt, last.get("response_text",""), last.get("psiguard_plan",{
                    "type":"max_attempts_exceeded","rule":"safety_catch"
                }), attempts, t0, return_debug)

            guided_prompt = current_prompt
            strategy_overrides = {}
            if attempts and self.health_config.recohere_enabled:
                thr = self.health_config.llm_max_kappa_tension
                if last_structural_E >= self.health_config.recohere_margin * thr:
                    guided = self.recoherence.preemptive(current_prompt)
                    guided_prompt = guided["prompt"]; strategy_overrides = guided["strategy"]
                    self.model.apply_strategy(strategy_overrides)

            # --- Generate + sigma ---
            if hasattr(self.model, "generate_stream"):
                ling, attn, lprob = [], [], []
                full_text = ""
                final_sample = None
                # Streamlit: Show token streaming
                message_placeholder = st.empty()
                streamed_text = ""
                
                try:
                    for full_text, sample in self.model.generate_stream(guided_prompt, max_new_tokens=120):
                        ling.append(self.rhetorical_detector._linguistic_sigma(self.rhetorical_detector.nlp(sample.token_str)))
                        attn.append(sample.attn_entropy_norm)
                        lprob.append(sample.logprob_entropy_norm)
                        final_sample = sample
                        streamed_text += sample.token_str
                        message_placeholder.markdown(streamed_text + " ▌") # Show streaming
                    message_placeholder.markdown(streamed_text) # Final text
                except Exception as e:
                    st.error(f"Error during streaming: {e}")
                    return self._fail_result(prompt, streamed_text, {"type":"generation_error", "reason": str(e)}, attempts, t0, return_debug)

                sigma_t = fuse_sigma(ling, attn, lprob)
                response_text = streamed_text # Use the text built from the stream
                if not response_text and prompt: # Handle empty response
                    response_text = " "
                confidence_scores = []
            else:
                # Fallback path
                response_text, confidence_scores = self.model.generate_with_confidence(guided_prompt, mode=mode)
                if not response_text and prompt: response_text = " "
                sigma_t = self.rhetorical_detector._linguistic_sigma(self.rhetorical_detector.nlp(response_text))

            attempt = {
                "attempt_id": uuid.uuid4().hex, "ts_iso": _now_iso(),
                "prompt_text_preview": guided_prompt[:160],
                "response_text": response_text,
                "confidence_scores": confidence_scores,
                "metrics": {},
                "psiguard_plan": None, "mode": mode,
            }

            structural = self.controller.run_inference_step({"E_hint": 0.25 + 0.5*(sigma_t/2.0)})
            last_structural_E = float(structural.get("metrics",{}).get("E",0.0))

            det = self.rhetorical_detector.detect_steering(response_text, json_output=False)
            kappa = float(det["kappa_t"]); Lambda = float(det["lambda_t"]); E = float(det["E_t"])
            attempt["metrics"] = {"sigma": sigma_t, "kappa": kappa, "lambda": Lambda, "E": E}

            thr = self.health_config.llm_max_kappa_tension
            scaled_thr = thr # No R-Boost since NGTS is disabled
            
            psiguard_plan = None
            if E >= scaled_T:
                psiguard_plan = {
                    "type":"structural_collapse","rule":"E_flux_exceeded",
                    "reason": f"E={E:.3f} ≥ {scaled_thr:.3f}",
                    "guardrail_deltas": {"temperature_multiplier":0.7, "top_p_multiplier":0.85}
                }
            attempt["psiguard_plan"] = psiguard_plan
            
            # Add attempt *after* metrics are calculated
            attempts.append(attempt)

            if psiguard_plan is None:
                return self._ok_result(prompt, response_text, attempts, t0, return_debug)

            if self.health_config.recohere_enabled and extra_recohere_attempts > 0:
                guided = self.recoherence.reactive(current_prompt, psiguard_plan)
                current_prompt = guided["prompt"] # Update prompt for next loop
                self.model.apply_strategy(guided["strategy"])
                extra_recohere_attempts -= 1
                continue

            if remaining_retries <= 0:
                return self._fail_result(prompt, response_text, psiguard_plan, attempts, t0, return_debug)
            remaining_retries -= 1
            
    def _ok_result(self, prompt, response, attempts, t0, dbg):
        return {"ok": True, "prompt": prompt, "response": response,
                "attempts": attempts if dbg else None, "latency_s": round(time.monotonic()-t0,3)}
    def _fail_result(self, prompt, response, breach, attempts, t0, dbg):
        return {"ok": False, "prompt": prompt, "response": response, "breach": breach,
                "attempts": attempts if dbg else None, "latency_s": round(time.monotonic()-t0,3)}


# ---
# --- PART 2: STREAMLIT APP UI & LOGIC ---
# ---

import pandas as pd
import plotly.graph_objects as go

# --- Caching: Load Models Only Once ---
# This is the most important part for performance.
@st.cache_resource
def load_models():
    """Loads all expensive models and objects."""
    print("--- LOADING MODELS (this should only run once) ---")
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("Spacy model not found. Downloading...")
        spacy.cli.download("en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")
        
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # --- HERE, we load the REAL adapter ---
    # Using gpt2 as it's small and fast for demos.
    # You can change "gpt2" to "gpt2-medium" or another model.
    model_adapter = HFWithAttentionAdapter(model_id="gpt2")
    
    rhetorical_detector = RhetoricalForceDetectorML(
        nlp_model=nlp_model,
        sentence_model=sentence_model,
        scorer=MockRhetoricScorer()
    )
    
    controller = ProductionPhase2Controller()
    reco_engine = RecoherenceEngine()
    
    # Load the main firewall class
    firewall = EnhancedIntegratedAICoherence(
        controller=controller,
        model_adapter=model_adapter,
        rhetorical_detector=rhetorical_detector,
        recoherence_engine=reco_engine,
        logger=print # Log to console
    )
    print("--- MODELS LOADED SUCCESSFULLY ---")
    return firewall

def create_gauge_chart(e_score, threshold):
    """Creates a Plotly gauge chart for the 'E' score (theme-agnostic)."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = e_score,
        title = {'text': "<b>Last Response Instability (E_t)</b>", 'font': {'size': 18}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'font': {'size': 28}},
        gauge = {
            'axis': {'range': [0, 1.2], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "#888888", 'thickness': 0.3},
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'rgba(0, 200, 0, 0.25)'},
                {'range': [threshold, 1.2], 'color': 'rgba(255, 0, 0, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# --- App Layout ---
st.set_page_config(page_title="AI Firewall Demo", layout="centered")
st.title("🤖 AI Safety Firewall Demo")
st.write("This demo is now using a **real Hugging Face model** (`gpt2`).")

# --- 1. THE SETTINGS MENU (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Firewall Settings")
    
    # Create a HealthConfig object from settings
    cfg = EnhancedHealthConfig()
    cfg.llm_max_kappa_tension = st.slider(
        "Safety Sensitivity (E_t Threshold)", 
        min_value=0.1, max_value=1.0, value=cfg.llm_max_kappa_tension, step=0.05,
        help="Lower = Stricter. This is the `llm_max_kappa_tension`."
    )
    cfg.recohere_enabled = st.toggle(
        "Enable Auto-Correction", value=cfg.recohere_enabled,
        help="If a response is 'bad', automatically retry. This is `recohere_enabled`."
    )
    
    st.divider()

    st.header("📊 Live Metrics")
    st.write("Shows the score of the *last* AI response.")
    gauge_placeholder = st.empty()
    
    if 'e_history' not in st.session_state:
        st.session_state.e_history = []
    
    st.write("Instability Score History (E_t)")
    history_chart = st.line_chart(st.session_state.e_history)

# --- Load the REAL firewall ---
# This call is cached, so it only runs on the first page load
try:
    firewall = load_models()
    # Update the firewall's config with the latest settings from the sidebar
    firewall.health_config = cfg
except Exception as e:
    st.error(f"Fatal Error Loading Models: {e}")
    st.exception(e)
    st.stop()


# --- 2. THE CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("🔬 Firewall Internals (View Details)"):
                st.json(message["details"])

# React to user input
if prompt := st.chat_input("Ask the AI something... (e.g., 'What is a dog?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 3. RUN THE REAL FIREWALL ---
    with st.chat_message("assistant"):
        # The streaming text will appear here
        result = firewall.generate_safely(prompt)
        
        # --- 4. DISPLAY THE FINAL RESULTS ---
        final_response_text = result.get("response", "Error: No response generated.")
        
        # The text has already been streamed, but we add it to history
        # We also create the "details" expander
        
        final_e_score = result.get("attempts", [{}])[-1].get("metrics", {}).get("E", 0)
        threshold = cfg.llm_max_kappa_tension
        
        if result["ok"]:
            details = {"status": "✅ ALLOWED", "attempts": result["attempts"]}
            st.session_state.messages.append({
                "role": "assistant", "content": final_response_text, "details": details
            })
        else:
            # If it was blocked, the streamed text might be the bad attempt
            # We show an error and log the details
            st.error("Firewall blocked this response. (See details below)")
            details = {"status": "❌ BLOCKED", "breach": result.get("breach"), "attempts": result.get("attempts")}
            st.session_state.messages.append({
                "role": "assistant", "content": f"*(Response blocked: {final_response_text})*", "details": details
            })

        # --- 5. UPDATE THE CHARTS ---
        fig = create_gauge_chart(final_e_score, threshold)
        gauge_placeholder.plotly_chart(fig, use_container_width=True)
        
        st.session_state.e_history.append(final_e_score)
        history_df = pd.DataFrame(st.session_state.e_history, columns=["E_t Score"])
        history_chart.add_rows(history_df.iloc[-1:])

# Draw the initial gauge on first run
if not st.session_state.messages:
    fig = create_gauge_chart(0, cfg.llm_max_kappa_tension)
    gauge_placeholder.plotly_chart(fig, use_container_width=True)