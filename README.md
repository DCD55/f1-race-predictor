# 🏎️ F1 Race Predictor & Analytics Dashboard

An interactive web application built with **Streamlit**, **Python**, and **Machine Learning** that analyzes historical telemetry, qualifying results, and driver performance to predict Grand Prix outcomes (Win and Podium probabilities).

🚀 **Live Demo:** [f1-race-predictor.streamlit.app](https://f1-race-predictor.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit%20Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---
<img width="1456" height="403" alt="Screenshot 2026-09-29 at 11 52 23 p m" src="https://github.com/user-attachments/assets/f7b497db-e76a-4462-8cd8-37003f593923" />
<img width="1456" height="752" alt="Screenshot 2026-09-29 at 11 52 06 p m" src="https://github.com/user-attachments/assets/8121eef7-e367-4e8d-b92c-0d1c29660b02" />

---


## ✨ Key Features
- **ML Predictive Models:** Trained pipelines evaluating driver recent form, starting grid positions, and constructor performance.
- **Dynamic UI/UX:** Custom-styled team badge cards featuring official branding colors, driver numbers, nationality flags, and F1 car assets.
- **Telemetry & Track Context:** Interactive official circuit maps and historical season/circuit statistics for every Grand Prix.

---

## 🛠️ Tech Stack
- **Frontend & UI:** Streamlit, Custom HTML/CSS
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Plotly

---

## 🚀 Getting Started Locally

1. **Clone the repository:**
   ```bash
   git clone git@github.com:DCD55/f1-race-predictor.git
   cd f1-race-predictor
2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the Streamlit app:
   streamlit run app.py

---

## 📚 Data Sources & Acknowledgments
* **Inspiration:** Inspired by and built upon concepts from the [2025 F1 Predictions project](https://github.com/mar-antaya/2025_f1_predictions) by Mar Antaya.
* **FastF1:** Telemetry, session results, and timing data are powered by the [FastF1 Python library](https://github.com/theOehrly/FastF1), which accesses official data feeds.
* **Ergast Developer API:** Historical race results, standings, and circuit data provided by the [Ergast Motor Racing API](http://ergast.com/mrd/).
* **Visual Assets:** Official team logos, car renders, and circuit layouts utilized for educational and analytical visualization purposes.


---

## 👨‍💻 Author
Developed by DCD55.
