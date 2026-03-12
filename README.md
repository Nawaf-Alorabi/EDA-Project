# EDA-Project
Unit3: EDA & visualization

## 📊 Traffic Accidents Analysis in Saudi Arabia (1437–1439)

This project performs **Exploratory Data Analysis (EDA)** on traffic accident data
across **16 cities in Saudi Arabia** over three Hijri years (1437, 1438, 1439).

### Project Structure

| File | Description |
|------|-------------|
| `EDA_Project_Code.ipynb` | Main Jupyter notebook with full EDA and analysis |
| `App.py` | Streamlit dashboard to interactively explore the results |
| `Injured_and_Dead_in_Accidents_*.csv` | Raw data files (years 1437, 1438, 1439) |

### 🚀 Running the Dashboard

1. **Install dependencies:**

```bash
pip install streamlit pandas numpy matplotlib seaborn
```

2. **Run the app:**

```bash
streamlit run App.py
```

3. Open the URL shown in the terminal (usually `http://localhost:8501`)

### Dashboard Sections

- **📋 Project Overview** — Summary metrics and yearly totals
- **🔍 Data Preview** — Interactive data table with year/city filters
- **📊 Descriptive Statistics** — Numerical summaries, missing values, and value counts
- **📈 Visualizations** — 8 interactive charts (trends, heatmaps, distributions, correlations)
- **💡 Key Insights** — Key findings from the analysis

### Dependencies

- Python 3.8+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
