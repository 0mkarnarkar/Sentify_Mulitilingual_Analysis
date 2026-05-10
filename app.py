import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from deep_translator import GoogleTranslator
from langdetect import detect
import time
import math
import re

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Sentify | Intelligence", page_icon="✨", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 0%, #111111 0%, #000000 70%);
        color: #EDEDED;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600; letter-spacing: -0.02em; }
    
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background-color: #FFFFFF;
        color: #000000; border: none; border-radius: 8px; 
        width: 100%; font-weight: 600; font-size: 15px; padding: 0.5rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { background-color: #E0E0E0; transform: translateY(-1px); }
    
    [data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 600; }
    [data-testid="stMetricLabel"] { color: #A1A1AA; }
    
    [data-testid="stSidebar"] { background-color: #0A0A0A; border-right: 1px solid #1F1F1F; }
    .stProgress .st-bo { background-color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. UNIVERSAL NLP ENGINE (FIXED TRANSLATION)
# ==========================================
class NLPAnalyticsEngine:
    def __init__(self):
        # Setting source to 'auto' handles any language Google supports
        self.translator = GoogleTranslator(source='auto', target='en')

    def process_single(self, text):
        text_str = str(text).strip()
        if not text_str: return None
            
        # 1. Safe Language Detection
        try:
            lang = detect(text_str).upper()
        except:
            lang = "AUTO"

        # 2. Bulletproof Translation
        try:
            en_text = self.translator.translate(text_str)
            if not en_text: en_text = text_str
        except Exception:
            en_text = text_str
            lang = "API_ERR"

        # 3. Sentiment Analytics
        blob = TextBlob(en_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity >= 0.05: label = "Positive" 
        elif polarity <= -0.05: label = "Negative" 
        else: label = "Neutral" 

        return {
            "Original": text_str, 
            "Language": lang, 
            "Translated": en_text,
            "Polarity": round(polarity, 4), 
            "Subjectivity": round(subjectivity, 4),
            "Label": label, 
            "WordCount": len(en_text.split())
        }

# ==========================================
# 3. BATCH PROCESSOR (LOAD BALANCER)
# ==========================================
class BatchProcessorQueue:
    def __init__(self, dataframe, text_col, batch_size=10):
        self.df = dataframe.dropna(subset=[text_col])
        self.text_col = text_col
        self.batch_size = batch_size
        self.engine = NLPAnalyticsEngine()
        self.total_rows = len(self.df)

    def execute_queue(self, progress_bar, status_text):
        results = []
        chunks = math.ceil(self.total_rows / self.batch_size)
        
        for i in range(chunks):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.total_rows)
            chunk_df = self.df.iloc[start_idx:end_idx]
            
            status_text.text(f"Processing chunk {i+1} of {chunks}...")
            
            for _, row in chunk_df.iterrows():
                analyzed = self.engine.process_single(row[self.text_col])
                if analyzed: results.append(analyzed)
            
            progress_bar.progress((i + 1) / chunks)
            time.sleep(0.5) 
            
        return pd.DataFrame(results)

# ==========================================
# 4. DASHBOARD UI & REPORT GENERATOR
# ==========================================
st.sidebar.title("Sentify ✨")
st.sidebar.markdown("Natural Language Intelligence")
st.sidebar.divider()
mode = st.sidebar.radio("Workspaces", ["Live Console", "Batch Processing"])
st.sidebar.divider()

engine = NLPAnalyticsEngine()
color_map = {"Positive": "#FFFFFF", "Negative": "#555555", "Neutral": "#888888"}

if mode == "Live Console":
    st.title("Live Console")
    st.markdown("<p style='color:#A1A1AA;'>Real-time multilingual sentiment decoding.</p>", unsafe_allow_html=True)
    
    user_input = st.text_area("Input Stream", height=150, placeholder="Enter text to analyze...")
    
    if st.button("Analyze Sequence"):
        if user_input:
            with st.spinner("Processing..."):
                res = engine.process_single(user_input)
                
                st.subheader("Output Metrics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Language", res['Language'])
                c2.metric("Classification", res['Label'])
                c3.metric("Polarity", res['Polarity'])
                c4.metric("Subjectivity", res['Subjectivity'])
                
                st.divider()
                if res['Language'] not in ['EN', 'AUTO'] or res['Original'] != res['Translated']:
                    st.info(f"**English Translation:** {res['Translated']}")

elif mode == "Batch Processing":
    st.title("Data Ingestion")
    st.markdown("<p style='color:#A1A1AA;'>Bulk process CSV, JSON, or TXT (line-by-line) datasets.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload File (.csv, .json, .txt)", type=["csv", "json", "jsonl", "txt"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode("utf-8")
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                df = pd.DataFrame(lines, columns=['Content'])
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                try: df = pd.read_json(uploaded_file)
                except: 
                    uploaded_file.seek(0)
                    df = pd.read_json(uploaded_file, lines=True)
                
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            with st.form("batch_config"):
                col1, col2 = st.columns(2)
                with col1: text_col = st.selectbox("Select Text Column", df.columns)
                with col2: limit = st.slider("Sample Size", 10, 500, 50)
                submit = st.form_submit_button("Initialize Batch")
            
            if submit:
                p_bar = st.progress(0)
                status = st.empty()
                
                queue = BatchProcessorQueue(df.head(limit), text_col, batch_size=10)
                final_df = queue.execute_queue(p_bar, status)
                
                status.success("Processing complete.")
                
                st.divider()
                st.subheader("Analytics Dashboard")
                
                # --- PIE CHART AND BAR CHART ---
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_pie = px.pie(
                        final_df, names='Label', 
                        title='Sentiment Distribution', hole=0.4,
                        color='Label', color_discrete_map=color_map
                    )
                    fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_chart2:
                    fig_bar = px.histogram(
                        final_df, x='Label', 
                        title='Sentiment Volume', 
                        color='Label', color_discrete_map=color_map
                    )
                    fig_bar.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    fig_bar.update_yaxes(title_text='Count')
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- 3D VISUALIZATION ---
                st.markdown("### 3D Sentiment Cluster")
                fig_3d = px.scatter_3d(
                    final_df, x='Polarity', y='Subjectivity', z='WordCount',
                    color='Label', color_discrete_map=color_map,
                    hover_name='Original'
                )
                fig_3d.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.dataframe(final_df, height=300)
                
                # --- EXPORT REPORT GENERATION ---
                csv = final_df.to_csv(index=False).encode('utf-8')
                
                html_pie = fig_pie.to_html(full_html=False, include_plotlyjs='cdn')
                html_bar = fig_bar.to_html(full_html=False, include_plotlyjs=False) 
                html_3d = fig_3d.to_html(full_html=False, include_plotlyjs=False)
                html_table = final_df[['Original', 'Label', 'Polarity', 'Language']].to_html(index=False, classes='data-table')
                
                html_report = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Sentify Intelligence Report</title>
                    <style>
                        body {{ font-family: 'Inter', sans-serif; background-color: #0A0A0A; color: #EDEDED; padding: 40px; margin: 0; }}
                        h1 {{ color: #FFFFFF; border-bottom: 1px solid #333; padding-bottom: 10px; }}
                        h2 {{ color: #A1A1AA; margin-top: 40px; }}
                        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                        .chart-container {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin-top: 20px; border: 1px solid rgba(255,255,255,0.1); }}
                        .data-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: rgba(255,255,255,0.02); }}
                        .data-table th, .data-table td {{ border: 1px solid #333; padding: 12px; text-align: left; }}
                        .data-table th {{ background-color: #1A1A1A; color: #FFF; }}
                        .footer {{ margin-top: 50px; font-size: 12px; color: #666; text-align: center; }}
                    </style>
                </head>
                <body>
                    <h1>✨ Sentify | Executive Intelligence Report</h1>
                    <p>Total Records Processed: {len(final_df)}</p>
                    
                    <h2>High-Level Analytics</h2>
                    <div class="grid">
                        <div class="chart-container">{html_pie}</div>
                        <div class="chart-container">{html_bar}</div>
                    </div>

                    <h2>Interactive 3D Sentiment Mapping</h2>
                    <div class="chart-container">
                        {html_3d}
                    </div>
                    
                    <h2>Processed Data Matrix</h2>
                    {html_table}
                    
                    <div class="footer">Generated securely by Sentify Analytics Engine.</div>
                </body>
                </html>
                """

                # --- EXPORT BUTTONS ---
                st.divider()
                st.write("### Export Options")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📄 Download Raw Data (CSV)", 
                        data=csv, 
                        file_name='sentify_data.csv',
                        mime='text/csv'
                    )
                
                with col_dl2:
                    st.download_button(
                        label="🌐 Download Interactive HTML Report", 
                        data=html_report, 
                        file_name='sentify_analytics_report.html',
                        mime='text/html'
                    )
                    
        except Exception as e:
            st.error(f"Error loading file: {e}")