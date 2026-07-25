import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import sqlite3
from datetime import datetime
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    return vectorizer, model

vectorizer, model = load_model()

# ==========================================
# DATABASE SETUP
# ==========================================
def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

# ==========================================
# TEXT CLEANING
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# LIME EXPLAINER — HARDCODED FIX
# ==========================================
@st.cache_resource
def get_explainer():
    return LimeTextExplainer(class_names=['REAL', 'FAKE'])

explainer = get_explainer()

def predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    return model.predict_proba(vec)

# ==========================================
# CSS — MODERN UI
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    
    h1, h2, h3, h4, h5, h6 { color: #1e293b !important; font-weight: 700; }
    p, label, div { color: #334155 !important; }
    
    .title { font-size: 3rem; font-weight: 800; text-align: center; color: #1e40af !important; letter-spacing: -1px; }
    .subtitle { text-align: center; color: #64748b !important; font-size: 1.2rem; font-weight: 400; }
    
    .stSidebar { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border-right: none; }
    .stSidebar h2, .stSidebar h3 { color: #f1f5f9 !important; }
    .stSidebar p, .stSidebar label, .stSidebar div { color: #cbd5e1 !important; }
    .stSidebar [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    .stSidebar [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important; border: none; border-radius: 16px;
        padding: 1rem 2rem; font-size: 1.15rem; font-weight: 700;
        width: 100%; letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(59,130,246,0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139,92,246,0.4);
    }
    
    .stTextArea textarea {
        border: 2px solid #e2e8f0; border-radius: 16px; background: #fff;
        color: #1e293b !important; min-height: 160px; font-size: 1rem;
        padding: 1rem; transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6; box-shadow: 0 0 0 4px rgba(59,130,246,0.1);
    }
    
    .result-fake {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #fca5a5; border-radius: 20px; padding: 2rem;
        text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220,38,38,0.1);
    }
    .result-fake h2 { color: #dc2626 !important; font-size: 2rem; font-weight: 800; }
    .result-fake h3 { color: #ef4444 !important; font-weight: 500; }
    
    .result-real {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac; border-radius: 20px; padding: 2rem;
        text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(22,163,74,0.1);
    }
    .result-real h2 { color: #16a34a !important; font-size: 2rem; font-weight: 800; }
    .result-real h3 { color: #22c55e !important; font-weight: 500; }
    
    [data-testid="stMetricValue"] { color: #1e293b !important; font-weight: 700; font-size: 1.5rem; }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.85rem; }
    
    .stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); border-radius: 10px; }
    
    .stAlert { border-radius: 12px; font-weight: 500; }
    .stWarning { background: #fffbeb; border: 1px solid #fde68a; }
    .stInfo { background: #eff6ff; border: 1px solid #bfdbfe; }
    .stError { background: #fef2f2; border: 1px solid #fecaca; }
    
    .footer { text-align: center; color: #94a3b8 !important; padding: 1.5rem; border-top: 1px solid #e2e8f0; font-size: 0.9rem; }
    .footer p { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered News Authenticity Check with Explainability</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 📊 Statistics")
    
    total = pd.read_sql("SELECT COUNT(*) as count FROM predictions", conn).iloc[0,0]
    fake_count = pd.read_sql("SELECT COUNT(*) as count FROM predictions WHERE prediction='FAKE'", conn).iloc[0,0]
    
    c1, c2 = st.columns(2)
    c1.metric("Total Checks", total)
    c2.metric("Fake Detected", fake_count)
    
    if total > 0:
        st.progress(fake_count/total, text="Fake Ratio")
    
    st.markdown("---")
    st.markdown("## 🤖 Model Info")
    st.markdown("**Technique:** TF-IDF + N-grams")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Accuracy:** 99.5%")
    st.markdown("---")
    st.warning("⚠️ This is an AI tool. Results are directional, not definitive.")

# ==========================================
# MAIN INPUT
# ==========================================
st.markdown("### 📝 Paste News Article")
news_text = st.text_area("", height=200, placeholder="Paste the news article here...", label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Analyze News", use_container_width=True):
    if not news_text.strip():
        st.error("Please paste a news article first!")
    elif len(news_text.split()) < 5:
        st.error("Article too short!")
    else:
        with st.spinner("Analyzing..."):
            words = news_text.split()
            if len(words) > 300:
                st.info(f"ℹ️ Analyzing first 300 words (article is {len(words)} words long)")
                text_to_analyze = ' '.join(words[:300])
            else:
                text_to_analyze = news_text
            
            cleaned = clean_text(text_to_analyze)
            vec = vectorizer.transform([cleaned])
            prob = model.predict_proba(vec)[0]
            
            classes = model.classes_
            if classes[0] == 'FAKE':
                fake_prob = prob[0]
                real_prob = prob[1]
            else:
                fake_prob = prob[1]
                real_prob = prob[0]
            
            if fake_prob > real_prob:
                prediction = "FAKE"
                confidence = fake_prob * 100
            else:
                prediction = "REAL"
                confidence = real_prob * 100
            
            # Save
            cursor = conn.cursor()
            cursor.execute('INSERT INTO predictions (news_text, prediction, confidence) VALUES (?, ?, ?)',
                         (news_text[:500], prediction, confidence))
            conn.commit()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction == "FAKE":
                st.markdown(f"""
                <div class="result-fake">
                    <h2>🔴 FAKE NEWS</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-real">
                    <h2>🟢 REAL NEWS</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📊 Detailed Scores"):
                c1, c2 = st.columns(2)
                c1.metric("FAKE Score", f"{fake_prob:.4f}")
                c2.metric("REAL Score", f"{real_prob:.4f}")
            
            # ==========================================
            # LIME EXPLANATION — HARDCODED FIX
            # ==========================================
            st.markdown("---")
            st.markdown("### 💡 Why This Prediction?")
            
            with st.spinner("Generating explanation..."):
                try:
                    if len(cleaned.split()) < 20:
                        st.info("📝 Text too short for detailed word explanation. Try a longer article (20+ words) to see word importance.")
                    else:
                        label_to_explain = 1
                        exp = explainer.explain_instance(
                            cleaned,
                            predict_proba,
                            num_features=8,
                            labels=(label_to_explain,)
                        )
                        fig = exp.as_pyplot_figure(label=label_to_explain)
                        plt.title("Word Importance (Red = FAKE, Green = REAL)", fontweight='bold')
                        st.pyplot(fig)
                except Exception as e:
                    st.info(f"📝 Could not generate detailed explanation for this text.")
            
            st.info("🔴 Red = FAKE indicators | 🟢 Green = REAL indicators")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>📰 Fake News Detector — Logistic Regression + TF-IDF + LIME</p>
    <p>© 2026 Fiza Aslam | Data Scientist</p>
</div>
""", unsafe_allow_html=True)