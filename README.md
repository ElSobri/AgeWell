````md
# AGEWELL: Evidence-Based Longevity Engine  
**"Age better with evidence."**

## Project Vision
Traditional health reports are static and siloed. Users receive blood work (ApoB, HbA1c, etc.) but lack an integrated framework to translate numbers into action. **AgeWell** acts as a **Clinical Reasoning Engine**, mimicking the logic of a senior longevity physician by cross-referencing individual biometrics against the latest peer-reviewed literature.

## Technical Architecture
The project utilizes a **Decoupled Architecture** to ensure high scalability and modular maintenance:

- **LLM Engine:** Gemini 2.5 Flash *(State-of-the-art 2026 Reasoning Model)*
- **Embedding Model:** BGE-M3 *(Multi-lingual, 1024-dimensional vectors for high-precision semantic retrieval)*
- **Vector Store:** FAISS *(Facebook AI Similarity Search)*
- **Frontend:** Streamlit *(Custom clinical UI with professional styling)*
- **Structure:** Modular separation of concerns between UI (`app.py`) and Reasoning Engine (`logic.py`)

## Knowledge Taxonomy (The 8 Longevity Pillars)
The AgeWell knowledge base is grounded in **45+ authoritative medical sources**, categorized into eight critical domains:

1. **Metabolism:** Insulin sensitivity, metabolic flexibility, and HbA1c management.  
2. **Cardiovascular:** Lipid profiles (ApoB/LDL), hypertension, and endothelial function.  
3. **Fitness:** VO2 Max mortality correlations and sarcopenia prevention through resistance training.  
4. **Nutrition:** Mediterranean protocols, Intermittent Fasting (TRF), and autophagy pathways.  
5. **Recovery:** Circadian rhythm optimization, sleep hygiene, and HRV (Heart Rate Variability).  
6. **Biomarkers:** Micronutrient density (Vitamin D/Omega-3) and chronic inflammation (CRP) monitoring.  
7. **Neurology:** Cognitive reserve, stress management (Cortisol), and neuroplasticity.  
8. **Cellular Longevity:** Biological age vs. chronological age, senescent cells, and mitochondrial health.  

## RAG Foundation Benchmarks
To ensure objective reasoning, the engine is embedded with five foundational clinical benchmarks:

- **Body Composition:** BMI ranges, body fat percentage standards, and Waist-to-Hip Ratio (WHR) predictors.  
- **Reference Ranges:** Comprehensive Metabolic Panel (CMP) and lipid target values.  
- **Vitals:** AHA/ACC Blood Pressure categories and resting heart rate (RHR) norms.  
- **Fitness Benchmarks:** VO2 Max percentiles and grip strength standards for longevity.  
- **Optimal Zones:** Defining optimal ranges for Vitamin D, B12, and Ferritin.  

## Key Engineering Innovations

### 1) Severity Override Logic
Designed to solve AI distraction. When critical outliers (e.g., BMI over 30) are detected, the system triggers a **priority-weighting override**, forcing the AI to address urgent **SHIFT protocols** before addressing missing or secondary lab data.

### 2) Deterministic Output Sanitization
To maintain a clinical-grade user experience, I developed a multi-stage post-processing pipeline. It strips inconsistent AI formatting and injects sanitized HTML to ensure the professional report remains visually perfect.

### 3) Recursive Semantic Chunking
Optimized retrieval using **800-character chunks** with a **150-character overlap**. This strategy ensures that clinical context and causal logic within medical documents remain intact during vectorization.

## Installation and Setup
Follow these four steps to run AgeWell locally:

### 1) Clone the repository and install dependencies
```bash
git clone https://github.com/yourusername/AgeWell.git
cd AgeWell
pip install -r requirements.txt
````

### 2) Configure environment variables (`.env`)

Create a file named `.env` in the root directory and add your API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 3) Initialize the clinical index (vectorization)

Place your medical documents in the `/knowledge_base` folder and run:

```bash
python build_index.py
```

### 4) Launch the application

Run the following command to start the web interface:

```bash
streamlit run app.py
```

## Medical Disclaimer

This tool is for educational and informational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare professional before making health decisions based on information provided by AgeWell.

## Project Maintenance

* **Author:** Wei Ting Chang
* **Stack:** Python, FAISS, Streamlit, Google Generative AI

---
