# 🚗 Traffic Accidents Analysis — Saudi Arabia (1437–1439 H)

A comprehensive Exploratory Data Analysis (EDA) of traffic accidents across 16 cities in Saudi Arabia, covering Hijri years 1437–1439.

## 📊 Dashboard

An interactive Streamlit dashboard (`App.py`) provides a professional, single-page view of the full analysis:

- **Project Overview** — Dataset summary and key metrics
- **Data Preview** — Tabbed view of injured and death datasets
- **Descriptive Statistics** — Statistical summaries with data quality checks
- **10 Visualizations** — Each follows a Question → Chart → Insights structure:
  1. Total Accidents Over Time (Line Chart)
  2. Monthly Accidents Heatmap
  3. Accidents by City (Horizontal Bar Chart)
  4. Age Group Distribution by City (Stacked Bar)
  5. Gender Distribution (Pie Charts)
  6. Saudi vs Non-Saudi by City (Grouped Bar)
  7. Inside vs Outside City (Box Plot)
  8. Correlation Heatmap
  9. Monthly Trend & Forecast (Line + Regression)
  10. City Risk Clustering (PCA Scatter)
- **ML Models** — Risk classification (Random Forest, Gradient Boosting, Logistic Regression) with ROC curves, cross-validation, and feature importances
- **K-Means Clustering** — Unsupervised city risk profiling with Elbow Method and PCA visualization

## 🚀 How to Run

```bash
# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

# Launch the dashboard
streamlit run App.py
```

## 📁 Project Structure

```
EDA-Project/
├── App.py                                   # Streamlit dashboard
├── Traffic_Accidents_KSA_AnalysisMain.ipynb # Original analysis notebook
├── README.md
├── injured/
│   ├── Injured in Accidents 1437 H.csv
│   ├── Injured in Accidents 1438 H.csv
│   └── Injured in Accidents 1439 H.csv
├── dead/
│   ├── Dead in Accidents 1437 H.csv
│   ├── Dead in Accidents 1438 H.csv
│   └── Dead in Accidents 1439 H.csv
└── excel/                                   # Excel source files
```

## 🛠️ Dependencies

- Python 3.8+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---
*Tuwaiq Data Science & AI Bootcamp*
