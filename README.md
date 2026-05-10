<div align="center">
  <img src="logo.jpeg" alt="Sentify Logo" width="150" style="border-radius: 20px;"/>
  <h1>Sentify | Multilingual Neural Analytics Engine</h1>
  <p><b>Enterprise-Grade Natural Language Processing and 3D Data Intelligence</b></p>
</div>

---

## Project Overview
Sentify is a high-performance NLP (Natural Language Processing) application designed to transform unstructured multilingual text into actionable business intelligence[cite: 1, 2]. [cite_start]The system acts as a "Neural Pipeline" that identifies, translates, categorizes, and visualizes emotional data in 3D space[cite: 3].

## Core Capabilities
* **Universal Ingestion:** Accepts data via a real-time Live Console, or batch processes CSV, JSON, and raw TXT (line-by-line) datasets[cite: 4].
* **Auto-Language Detection:** Identifies over 130 languages using statistical linguistic profiles[cite: 5].
* **Neural Translation:** Standardizes all regional inputs into English using a Google-backed translation API to ensure consistent analytical scoring[cite: 6].
* **Dual-Metric Analytics:** Calculates precise mathematical scores for both Polarity (emotional tone from -1.0 to +1.0) and Subjectivity (data type from 0.0 fact to 1.0 opinion)[cite: 7, 8].
* **3D Neural Clustering:** Visualizes data points across three axes (Polarity, Subjectivity, and Word Count) using interactive Plotly 3D scatter plots[cite: 9].
* **Portable Reporting:** Generates downloadable, standalone Interactive HTML Reports featuring embedded 3D models and data distribution charts[cite: 10].

## Advanced Architecture
* **Smart Queue Load Balancer:** Sentify implements a custom Batch Processor Queue that breaks large datasets into smaller chunks[cite: 11]. [cite_start]This prevents API throttling and ensures absolute system stability during heavy data ingestion[cite: 12].
* **Spatial Data Topology:** By mapping "Subjectivity" alongside sentiment, the system allows analysts to visually distinguish between objective, factual reports (e.g., bug reports) and subjective, emotional vents—a critical feature for real-world customer support routing[cite: 13, 14, 15, 16, 17].

## 🛠️ The Technology Stack
* **Core:** Python 3.13 [cite: 11]
* **UI Framework:** Streamlit (Custom Dark Mode UI) [cite: 11]
* **NLP & Translation:** TextBlob, Deep-Translator, Langdetect [cite: 11]
* **Data Visualization:** Plotly [cite: 11]
* **Data Handling:** Pandas [cite: 11]

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Sentify_Multilingual_Analysis.git](https://github.com/yourusername/Sentify_Multilingual_Analysis.git)
   cd Sentify_Multilingual_Analysis

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
(Note: The environment requires Streamlit, TextBlob, Deep-Translator, Langdetect, Pandas, and Plotly.)

3. **Launch the application:**
   ```bash
   streamlit run app.py
