"""
App.py — Streamlit Dashboard for Traffic Accidents EDA Project
===============================================================
Saudi Arabia Traffic Accidents Analysis (1437–1439 H)
Tuwaiq Data Science & AI Bootcamp

Single-page scrolling dashboard with:
  - Project Overview
  - Data Preview & Descriptive Statistics
  - 10 Visualizations (Question → Chart → Insights)
  - 2 Model Sections (ML Classification + K-Means Clustering)

Run:  streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Accidents — Saudi Arabia",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS for polished styling
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem; color: #6c757d; text-align: center;
        margin-bottom: 2rem; font-weight: 400;
    }
    /* Section headers */
    .section-header {
        font-size: 1.7rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 3px solid #0f3460; padding-bottom: 0.4rem;
        margin-top: 3rem; margin-bottom: 1.2rem;
    }
    /* Viz question box */
    .viz-question {
        background: linear-gradient(135deg, #e8f4f8, #f0f7ff);
        border-left: 4px solid #2980B9; border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
        font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
    }
    /* Insight box */
    .viz-insight {
        background: linear-gradient(135deg, #f0faf0, #e8f8e8);
        border-left: 4px solid #27ae60; border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem; margin-top: 0.8rem; margin-bottom: 2rem;
        font-size: 0.95rem; color: #2c3e50;
    }
    /* Model explanation box */
    .model-box {
        background: linear-gradient(135deg, #faf0ff, #f5e8ff);
        border-left: 4px solid #8E44AD; border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
        font-size: 0.95rem; color: #2c3e50;
    }
    /* Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 2.5rem 0;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data Loading & Cleaning (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """
    Replicate the cleaning function from the notebook.
    Loads 6 CSV files (3 injured + 3 dead), cleans, and merges.
    """
    base = os.path.dirname(os.path.abspath(__file__))

    # Month name mapping (Arabic → number), including alternate spellings
    month_map = {
        'محرم': 1, 'صفر': 2, 'ربيع أول': 3, 'ربيع ثانى': 4,
        'جمادى أول': 5, 'جمادى ثانى': 6, 'رجب': 7, 'شعبان': 8,
        'رمضان': 9, 'شوال': 10, 'ذى القعدة': 11, 'ذى الحجة': 12,
        'مــحــر م': 1, 'صــــفـر': 2, 'ر جـــب': 7,
    }

    # City names assigned in order of appearance in each file
    cities = [
        'Riyadh', 'Madinah', 'Al-Sharqiyah', 'Northern Borders',
        'Tabuk', 'Al-Jouf', 'Hail', 'Najran', 'Al-Qassim', 'Al-Baha',
        'Asir', 'Jazan', 'Jeddah', 'Taif', 'Makkah', 'Al-Qurayyat',
    ]

    columns = [
        'month', 'male', 'female', 'total_gender',
        'inside_city', 'outside_city', 'total_location',
        'under_18', 'age_18_30', 'age_30_40', 'age_40_50', 'age_50_plus', 'total_age',
        'saudi', 'non_saudi', 'total_nationality',
    ]

    def clean(df, year, data_type):
        df.columns = df.iloc[1]
        df = df.iloc[2:].reset_index(drop=True)
        df.columns = columns
        df = df[~df.apply(lambda row: row.astype(str).str.contains('عدد').any(), axis=1)]
        df['month'] = df['month'].map(month_map)
        df = df.dropna()
        df['month'] = df['month'].astype(int)
        df = df.iloc[:-12]  # remove yearly totals
        df['year'] = year
        df['injured'] = 1 if data_type == 'injured' else 0
        df['dead'] = 1 if data_type == 'dead' else 0
        cols_to_drop = [c for c in df.columns if 'total' in c]
        df = df.drop(columns=cols_to_drop)
        city_col = []
        for city in cities:
            city_col.extend([city] * 12)
        df['city'] = city_col[:len(df)]
        return df

    # Load all 6 files
    inj_files = [
        (os.path.join(base, 'injured', f'Injured in Accidents {y} H.csv'), y)
        for y in [1437, 1438, 1439]
    ]
    ded_files = [
        (os.path.join(base, 'dead', f'Dead in Accidents {y} H.csv'), y)
        for y in [1437, 1438, 1439]
    ]

    frames = []
    for path, year in inj_files:
        raw = pd.read_csv(path, engine='python', encoding='utf-8-sig', header=None)
        frames.append(clean(raw, year, 'injured'))
    for path, year in ded_files:
        raw = pd.read_csv(path, engine='python', encoding='utf-8-sig', header=None, sep=';')
        frames.append(clean(raw, year, 'dead'))

    df = pd.concat(frames, ignore_index=True)

    num_cols = ['male', 'female', 'inside_city', 'outside_city',
                'under_18', 'age_18_30', 'age_30_40', 'age_40_50',
                'age_50_plus', 'saudi', 'non_saudi']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    inj = df[df['injured'] == 1].copy()
    ded = df[df['dead'] == 1].copy()

    return df, inj, ded, num_cols


# Month labels for charts
MONTH_LABELS = [
    'Muharram', 'Safar', "Rabi' I", "Rabi' II",
    'Jumada I', 'Jumada II', 'Rajab', "Sha'ban",
    'Ramadan', 'Shawwal', "Dhul-Qa'da", 'Dhul-Hijja',
]

# Load data
df, inj, ded, NUM_COLS = load_and_clean_data()

# Chart styling
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.facecolor': '#FAFAFA', 'axes.facecolor': '#FAFAFA'})


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🚗 Traffic Accidents Analysis in Saudi Arabia</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hijri Years 1437 – 1439 &nbsp;|&nbsp; Tuwaiq Data Science & AI Bootcamp</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Project Overview</div>', unsafe_allow_html=True)

col_intro1, col_intro2 = st.columns([3, 2])
with col_intro1:
    st.markdown("""
    This project analyzes traffic accident data across **16 cities** in Saudi Arabia
    over **three Hijri years** (1437, 1438, 1439). The dataset includes monthly records
    of both **injured** and **deceased** individuals, with breakdowns by:

    - **Gender** (Male / Female)
    - **Age Group** (Under 18, 18–30, 30–40, 40–50, 50+)
    - **Nationality** (Saudi / Non-Saudi)
    - **Location** (Inside City / Outside City)

    **Objectives:**
    - Explore temporal and geographic patterns in accident data
    - Identify high-risk cities and demographic groups
    - Build ML models to predict high-risk scenarios
    - Cluster cities by risk profile for targeted safety policies
    """)

with col_intro2:
    st.markdown("#### Dataset Summary")
    st.metric("Total Records", f"{len(df):,}")
    c1, c2 = st.columns(2)
    c1.metric("Injured Records", f"{len(inj):,}")
    c2.metric("Death Records", f"{len(ded):,}")
    c3, c4 = st.columns(2)
    c3.metric("Cities", f"{df['city'].nunique()}")
    c4.metric("Years Covered", "3 (1437–1439)")
    st.metric("Total Injured", f"{int(inj[NUM_COLS].sum().sum()):,}")
    st.metric("Total Deaths", f"{int(ded[NUM_COLS].sum().sum()):,}")


# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATA PREVIEW
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔍 Data Preview</div>', unsafe_allow_html=True)

with st.expander("View Raw Data Sample", expanded=False):
    tab1, tab2 = st.tabs(["Injured Data", "Deaths Data"])
    with tab1:
        st.dataframe(inj.head(20), use_container_width=True, height=300)
    with tab2:
        st.dataframe(ded.head(20), use_container_width=True, height=300)
    st.caption(f"Combined dataset: **{len(df):,}** rows × **{df.shape[1]}** columns")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Descriptive Statistics</div>', unsafe_allow_html=True)

stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.markdown("#### Injured — Numerical Summary")
    st.dataframe(inj[NUM_COLS].describe().round(1).T, use_container_width=True)

with stat_col2:
    st.markdown("#### Deaths — Numerical Summary")
    st.dataframe(ded[NUM_COLS].describe().round(1).T, use_container_width=True)

st.markdown("")

val_c1, val_c2, val_c3 = st.columns(3)
with val_c1:
    st.markdown("**Records per Year**")
    yr_vc = df.groupby('year').size().reset_index(name='Records')
    yr_vc['year'] = yr_vc['year'].astype(int)
    st.dataframe(yr_vc, hide_index=True, use_container_width=True)
with val_c2:
    st.markdown("**Records per City (Top 8)**")
    city_vc = df['city'].value_counts().head(8).reset_index()
    city_vc.columns = ['City', 'Records']
    st.dataframe(city_vc, hide_index=True, use_container_width=True)
with val_c3:
    st.markdown("**Data Quality**")
    missing = df.isnull().sum().sum()
    st.success(f"Missing values: **{missing}**")
    st.info(f"Data types: {df[NUM_COLS].dtypes.unique()}")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">📈 Visualizations & Analysis</div>', unsafe_allow_html=True)

# ─────────────────────────────────────
# VIZ 1: Total Accidents Over Time
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ Are traffic accidents increasing or decreasing over the years?</div>', unsafe_allow_html=True)

yr_inj = inj.groupby('year')[NUM_COLS].sum().sum(axis=1)
yr_ded = ded.groupby('year')[NUM_COLS].sum().sum(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(yr_inj.index, yr_inj.values, 'o-', color='#2980B9', lw=2.5, ms=9,
        label='Injured', markerfacecolor='white', markeredgewidth=2.5)
ax.plot(yr_ded.index, yr_ded.values, 's--', color='#C0392B', lw=2.5, ms=9,
        label='Deaths', markerfacecolor='white', markeredgewidth=2.5)
for x, yi, yd in zip(yr_inj.index, yr_inj.values, yr_ded.values):
    ax.annotate(f'{int(yi):,}', (x, yi), textcoords='offset points',
                xytext=(0, 14), ha='center', fontsize=11, color='#2980B9', fontweight='bold')
    ax.annotate(f'{int(yd):,}', (x, yd), textcoords='offset points',
                xytext=(0, -18), ha='center', fontsize=11, color='#C0392B', fontweight='bold')
ax.fill_between(yr_inj.index, yr_inj.values, alpha=0.06, color='#2980B9')
ax.fill_between(yr_ded.index, yr_ded.values, alpha=0.06, color='#C0392B')
ax.set_xticks(yr_inj.index)
ax.set_xticklabels(['1437 H', '1438 H', '1439 H'], fontsize=12)
ax.set_title('Total Accidents Over Time (1437–1439 H)', fontsize=14, fontweight='bold', pad=14)
ax.set_ylabel('Total Cases')
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Gradual decline</strong> in injuries from 1437 to 1439, indicating improved road safety measures.<br>
• Deaths did not decrease at the same rate — suggesting accidents became <strong>fewer but more severe</strong>.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 2: Monthly Accidents Heatmap
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ Which months are the most dangerous for traffic accidents?</div>', unsafe_allow_html=True)

pivot = inj.groupby(['year', 'month'])[NUM_COLS].sum().sum(axis=1).unstack()
pivot.columns = MONTH_LABELS

fig, ax = plt.subplots(figsize=(15, 3.8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Total Injured'}, ax=ax)
ax.set_title('Monthly Injuries Heatmap by Year', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Hijri Year')
ax.set_yticklabels(['1437 H', '1438 H', '1439 H'], rotation=0)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Ramadan (Month 9)</strong> is consistently the most dangerous month across all three years.<br>
• Rajab and Sha'ban (months 7–8) also show elevated accident counts — the pre-Ramadan period.<br>
• Winter months (Muharram, Safar) generally record the fewest accidents.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 3: Accidents by City
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ Which cities have the highest number of accidents?</div>', unsafe_allow_html=True)

city_inj = inj.groupby('city')[NUM_COLS].sum().sum(axis=1).sort_values()
city_ded = ded.groupby('city')[NUM_COLS].sum().sum(axis=1)
fat_rate = city_ded / (city_inj + city_ded) * 100
colors = ['#C0392B' if fat_rate.get(c, 0) > fat_rate.median() else '#2980B9' for c in city_inj.index]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(city_inj.index, city_inj.values, color=colors, edgecolor='white', height=0.7)
for bar, v in zip(bars, city_inj.values):
    ax.text(v + 150, bar.get_y() + bar.get_height() / 2, f'{int(v):,}', va='center', fontsize=9)
ax.axvline(city_inj.median(), color='#2C3E50', lw=1.4, ls='--', label=f'Median  {int(city_inj.median()):,}')
ax.set_title('Total Injuries by City — 1437–1439 H\n(red = fatality rate above median)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Total Injured Cases')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Taif</strong> leads in total injury count, followed by Al-Sharqiyah and Riyadh.<br>
• <strong>Al-Baha and Tabuk</strong> have fatality rates above the median despite lower injury counts — their accidents are <strong>more fatal</strong> (mountainous/highway roads).<br>
• Cities marked in red require targeted safety interventions.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 4: Age Group Distribution by City
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ How do age groups differ across cities in terms of accident involvement?</div>', unsafe_allow_html=True)

age_cols = ['under_18', 'age_18_30', 'age_30_40', 'age_40_50', 'age_50_plus']
age_labels = ['< 18', '18–30', '30–40', '40–50', '50+']
AGE_COLORS = ['#1ABC9C', '#2980B9', '#8E44AD', '#E67E22', '#C0392B']

top12 = (inj.groupby('city')[NUM_COLS].sum().sum(axis=1)
         .sort_values(ascending=False).head(12).index)
city_age = inj.groupby('city')[age_cols].sum().loc[top12]
city_age.columns = age_labels

fig, ax = plt.subplots(figsize=(13, 6))
city_age.plot(kind='bar', stacked=True, color=AGE_COLORS, ax=ax,
              edgecolor='white', linewidth=0.5, width=0.75)
ax.set_title('Age Group Distribution by City (Top 12)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('City')
ax.set_ylabel('Total Injured')
ax.set_xticklabels(top12, rotation=35, ha='right', fontsize=9)
ax.legend(title='Age Group', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• The <strong>18–30 age group</strong> dominates accident involvement in all cities — the highest-risk demographic.<br>
• The <strong>30–40 age group</strong> ranks second — the working-age population most vulnerable to road incidents.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 5: Gender Distribution — Pie Chart
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ What is the gender split among injured and deceased individuals?</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, (dataset, title) in zip(axes, [(inj, 'Injured'), (ded, 'Deaths')]):
    m = dataset['male'].sum()
    f = dataset['female'].sum()
    wedges, texts, autotexts = ax.pie(
        [m, f], labels=['Male', 'Female'], colors=['#2980B9', '#E91E63'],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12}, explode=(0.04, 0.04))
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight('bold')
    ax.set_title(f'Gender Split — {title}\n({int(m+f):,} total)', fontsize=13, fontweight='bold', pad=14)
plt.suptitle('Gender Distribution: Injured vs Deaths', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Males</strong> represent ~83% of injured and ~87% of deceased — the overwhelmingly affected gender.<br>
• Road safety campaigns should <strong>primarily target males, especially in the 18–30 age group</strong>.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 6: Saudi vs Non-Saudi by City
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ How does nationality (Saudi vs Non-Saudi) affect accident distribution across cities?</div>', unsafe_allow_html=True)

city_saudi = inj.groupby('city')['saudi'].sum().loc[top12]
city_nonsaudi = inj.groupby('city')['non_saudi'].sum().loc[top12]

x = np.arange(len(top12))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 5.5))
b1 = ax.bar(x - w/2, city_saudi.values, w, color='#1ABC9C', label='Saudi', alpha=0.88, edgecolor='white')
b2 = ax.bar(x + w/2, city_nonsaudi.values, w, color='#8E44AD', label='Non-Saudi', alpha=0.88, edgecolor='white')
for b in [b1, b2]:
    for bar in b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+40,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(top12, rotation=35, ha='right', fontsize=9)
ax.set_title('Saudi vs Non-Saudi Injured — Top 12 Cities', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Total Injured')
ax.legend(fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Makkah, Madinah, and Jeddah</strong> show disproportionately high non-Saudi involvement — driven by Hajj/Umrah seasons and expatriate populations.<br>
• Interior cities like <strong>Hail and Al-Qassim</strong> are overwhelmingly Saudi — reflecting local traffic patterns.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 7: Inside vs Outside City — Box Plot
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ Are accidents outside city limits more severe than those inside?</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(8, 5.5))
bp = ax.boxplot(
    [inj['inside_city'].dropna(), inj['outside_city'].dropna()],
    labels=['Inside City', 'Outside City'], patch_artist=True,
    medianprops={'color': '#2C3E50', 'linewidth': 2.5},
    whiskerprops={'linewidth': 1.5}, capprops={'linewidth': 1.5},
    flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5})
for patch, col in zip(bp['boxes'], ['#2980B9', '#E67E22']):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
ax.set_title('Accident Distribution: Inside vs Outside City', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Cases per City-Month Record')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Outside-city accidents</strong> show greater variability and higher extreme values — <strong>highways are more dangerous</strong> and less predictable.<br>
• Outlier points in the outside-city category represent exceptional months like <strong>Ramadan and Hajj season</strong>.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 8: Correlation Heatmap
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ What are the relationships between different accident variables?</div>', unsafe_allow_html=True)

corr = inj[NUM_COLS + ['month', 'year']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation Heatmap — Injured Dataset', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Strong positive correlation</strong> between male counts and overall totals — expected given male dominance in accidents.<br>
• <strong>Month and year</strong> show low correlation with other variables — temporal patterns require deeper analysis.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 9: Monthly Trend & Forecast
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ Can we forecast accident trends for the next year (1440 H)?</div>', unsafe_allow_html=True)

year_palette = {1437: '#2980B9', 1438: '#1ABC9C', 1439: '#8E44AD'}

fig, ax = plt.subplots(figsize=(13, 5.5))
for yr in [1437, 1438, 1439]:
    mo = inj[inj['year'] == yr].groupby('month')[NUM_COLS].sum().sum(axis=1)
    ax.plot(mo.index, mo.values, 'o-', color=year_palette[yr], lw=2, ms=6,
            label=f'{yr} H', markerfacecolor='white', markeredgewidth=1.8)

# Linear Regression forecast
inj_copy = inj.copy()
inj_copy['time_idx'] = (inj_copy['year'] - 1437) * 12 + inj_copy['month']
inj_copy['period_total'] = inj_copy[NUM_COLS].sum(axis=1)
agg = inj_copy.groupby('time_idx')['period_total'].sum().reset_index()

lr = LinearRegression()
lr.fit(agg[['time_idx']], agg['period_total'])
future_idx = np.arange(37, 49).reshape(-1, 1)
forecast = lr.predict(future_idx)

ax.plot(range(1, 13), forecast, 'r--', lw=2, alpha=0.7, label='1440 H Forecast')
ax.fill_between(range(1, 13), forecast * 0.90, forecast * 1.10,
                color='red', alpha=0.07, label='±10% Error Band')
ax.axvspan(8.5, 9.5, alpha=0.1, color='#E67E22')
ax.text(9.0, ax.get_ylim()[1] * 0.96, 'Ramadan', ha='center', fontsize=9, color='#E67E22')
ax.set_xticks(range(1, 13))
ax.set_xticklabels([m[:5] for m in MONTH_LABELS], rotation=30, fontsize=9)
ax.set_title('Monthly Injuries Trend & 1440 H Forecast', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Total Injured (All Cities)')
ax.legend(fontsize=9, ncol=2)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

slope = lr.coef_[0]
st.markdown(f"""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• The forecast suggests a <strong>continued downward trend</strong> in injuries for 1440 H (slope: {slope:.1f} cases/month).<br>
• <strong>Ramadan</strong> remains the highest-risk month in every year.<br>
• The ±10% error band acknowledges this is a preliminary forecast requiring 1440 data for validation.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────
# VIZ 10: City Risk Clustering
# ─────────────────────────────────────
st.markdown('<div class="viz-question">❓ Can cities be grouped by their accident risk profiles?</div>', unsafe_allow_html=True)

# Build merged injured+dead table
inj_g = inj.groupby(['city', 'year', 'month'])[NUM_COLS].sum().add_suffix('_inj').reset_index()
ded_g = ded.groupby(['city', 'year', 'month'])[NUM_COLS].sum().add_suffix('_ded').reset_index()
ml = pd.merge(inj_g, ded_g, on=['city', 'year', 'month'])
ml['total_injured'] = ml[[c for c in ml.columns if '_inj' in c]].sum(axis=1)
ml['total_dead'] = ml[[c for c in ml.columns if '_ded' in c]].sum(axis=1)
ml['fat_rate'] = ml['total_dead'] / (ml['total_injured'] + ml['total_dead'] + 1e-9)

km_features = ['total_injured', 'total_dead', 'fat_rate',
               'outside_city_inj', 'outside_city_ded',
               'age_18_30_inj', 'age_30_40_inj']
X_km = StandardScaler().fit_transform(ml[km_features].fillna(0))
km = KMeans(n_clusters=4, random_state=42, n_init=10)
ml['cluster'] = km.fit_predict(X_km)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_km)
ml['pc1'] = coords[:, 0]
ml['pc2'] = coords[:, 1]

mean_fat = ml.groupby('cluster')['fat_rate'].mean().sort_values(ascending=False)
label_map = {old: new for new, old in enumerate(mean_fat.index)}
ml['cluster_label'] = ml['cluster'].map(label_map)

CLUSTER_COLORS = {0: '#C0392B', 1: '#E67E22', 2: '#2980B9', 3: '#1ABC9C'}

fig, ax = plt.subplots(figsize=(10, 7))
for c in sorted(ml['cluster_label'].unique()):
    sub = ml[ml['cluster_label'] == c]
    ax.scatter(sub['pc1'], sub['pc2'], color=CLUSTER_COLORS[c], alpha=0.65, s=50,
               edgecolors='white', lw=0.4,
               label=f'Cluster {c}  (fatality={sub["fat_rate"].mean()*100:.1f}%,  n={len(sub)})')
ax.set_title('City Risk Clustering — K-Means (k=4)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)')
ax.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Insights:</strong><br>
• <strong>Cluster 0 (Red):</strong> High fatality rate, low injury count — small cities with deadly highway accidents.<br>
• <strong>Cluster 1 (Orange):</strong> High death volumes — major cities during peak months.<br>
• <strong>Cluster 2 (Blue):</strong> Lowest risk — quiet cities or low-accident months.<br>
• <strong>Cluster 3 (Green):</strong> Moderate risk tied to major highway corridors.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 5 — MODELS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────
# MODEL 1: Classification (Risk Prediction)
# ────────────────────────────────────────────
st.markdown("### Model 1 — Risk Classification")

st.markdown("""<div class="model-box">
<strong>📖 Explanation:</strong><br>
We train three classification models to predict whether a city-month-year record will be
<strong>high-risk</strong> (top 33% fatality rate). The models learn from demographic and geographic
features of each record.<br><br>
<strong>Models used:</strong><br>
• <strong>Random Forest</strong> — 200 decision trees that vote (majority wins)<br>
• <strong>Gradient Boosting</strong> — Sequential trees that learn from each other's mistakes<br>
• <strong>Logistic Regression</strong> — A mathematical equation computing risk probability<br><br>
<strong>Evaluation metric:</strong> AUC (Area Under ROC Curve) — 1.0 = perfect, 0.5 = random guess
</div>""", unsafe_allow_html=True)

# Prepare ML data
le = LabelEncoder()
ml['city_enc'] = le.fit_transform(ml['city'])
ml['is_ramadan'] = (ml['month'] == 9).astype(int)
threshold = ml['fat_rate'].quantile(0.67)
ml['high_risk'] = (ml['fat_rate'] >= threshold).astype(int)

FEATURES = [
    'city_enc', 'month', 'year', 'is_ramadan',
    'male_inj', 'female_inj', 'inside_city_inj', 'outside_city_inj',
    'under_18_inj', 'age_18_30_inj', 'age_30_40_inj', 'age_40_50_inj', 'age_50_plus_inj',
    'saudi_inj', 'non_saudi_inj', 'outside_city_ded', 'age_18_30_ded', 'age_30_40_ded',
]

X = ml[FEATURES].fillna(0)
y = ml['high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}
MODEL_COLORS = ['#2980B9', '#1ABC9C', '#8E44AD']

results = {}
for name, model in MODELS.items():
    scaled = (name == 'Logistic Regression')
    Xtr, Xte = (X_tr_s, X_te_s) if scaled else (X_train, X_test)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring='roc_auc')
    fpr, tpr, _ = roc_curve(y_test, proba)
    results[name] = {
        'model': model, 'preds': preds, 'proba': proba,
        'auc': roc_auc_score(y_test, proba),
        'cm': confusion_matrix(y_test, preds),
        'fpr': fpr, 'tpr': tpr, 'cv': cv,
    }

# Display: ROC + CV + Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for (name, res), col in zip(results.items(), MODEL_COLORS):
    ax.plot(res['fpr'], res['tpr'], lw=2, color=col, label=f'{name}\nAUC={res["auc"]:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Guess')
ax.fill_between(results['Random Forest']['fpr'], results['Random Forest']['tpr'], alpha=0.06, color='#2980B9')
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curves — All Models')
ax.legend(fontsize=8, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
cv_means = [results[n]['cv'].mean() for n in MODELS]
cv_stds = [results[n]['cv'].std() for n in MODELS]
bars = ax.bar(list(MODELS.keys()), cv_means, color=MODEL_COLORS, alpha=0.85,
              edgecolor='white', yerr=cv_stds, capsize=5)
for b, v in zip(bars, cv_means):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Mean AUC (5-Fold CV)')
ax.set_title('Cross-Validation AUC Comparison')
ax.set_xticklabels(list(MODELS.keys()), rotation=12, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[2]
best = max(results, key=lambda n: results[n]['auc'])
disp = ConfusionMatrixDisplay(results[best]['cm'], display_labels=['Low Risk', 'High Risk'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix\n({best})')

plt.suptitle('ML Model Evaluation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Results table
st.markdown("#### Model Results")
res_df = pd.DataFrame({
    'Model': list(MODELS.keys()),
    'Test AUC': [f"{results[n]['auc']:.3f}" for n in MODELS],
    'CV Mean AUC': [f"{results[n]['cv'].mean():.3f}" for n in MODELS],
    'CV Std': [f"±{results[n]['cv'].std():.3f}" for n in MODELS],
})
st.dataframe(res_df, hide_index=True, use_container_width=True)

st.markdown(f"""<div class="viz-insight">
<strong>💡 Interpretation:</strong><br>
• <strong>{best}</strong> achieved the highest AUC ({results[best]['auc']:.3f}) — strong predictive performance.<br>
• All models surpassed <strong>AUC > 0.88</strong>, indicating the data is clean and patterns are learnable.<br>
• The narrow gap between models suggests the relationship between features and risk is <strong>fairly linear</strong>.
</div>""", unsafe_allow_html=True)

# Feature Importance
st.markdown("#### Feature Importances — Random Forest")

rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = ['#C0392B' if ('ded' in i or 'outside' in i) else '#2980B9' for i in fi.index]
ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='white', alpha=0.87)
ax.set_title('Feature Importances — Random Forest\n(red = death/outside-city related)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Importance Score')
for i, v in enumerate(fi.values):
    ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""<div class="viz-insight">
<strong>💡 Interpretation:</strong><br>
• <strong>Outside-city deaths</strong> is the most influential predictor — highway accidents are the primary risk factor.<br>
• <strong>Deaths in the 18–30 age group</strong> ranks second — young adults most prone to fatal accidents.<br>
• <strong>City identity</strong> matters — each city has unique risk characteristics beyond just numbers.
</div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# MODEL 2: K-Means Clustering
# ────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### Model 2 — K-Means Clustering")

st.markdown("""<div class="model-box">
<strong>📖 Explanation:</strong><br>
Unlike the classification model above (supervised learning with labeled outcomes),
<strong>K-Means clustering is unsupervised</strong> — it discovers natural groupings in the data
without predefined answers.<br><br>
<strong>The question is:</strong> What natural risk groups exist among city-month records?<br>
<strong>Method:</strong> We use the <strong>Elbow Method</strong> to find the optimal number of clusters (k=4),
then visualize the groups using <strong>PCA</strong> (Principal Component Analysis).
</div>""", unsafe_allow_html=True)

# Elbow Method
inertias = []
K_range = range(2, 10)
for k in K_range:
    km_e = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_e.fit(X_km)
    inertias.append(km_e.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(K_range, inertias, 'o-', color='#2980B9', lw=2.2, ms=7,
        markerfacecolor='white', markeredgewidth=2)
ax.axvline(4, color='#C0392B', lw=1.5, ls='--', label='Chosen k=4')
ax.set(xlabel='Number of Clusters (k)', ylabel='Inertia',
       title='Elbow Method — Optimal k Selection')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Cluster scatter (reuse from above)
fig, ax = plt.subplots(figsize=(10, 7))
for c in sorted(ml['cluster_label'].unique()):
    sub = ml[ml['cluster_label'] == c]
    ax.scatter(sub['pc1'], sub['pc2'], color=CLUSTER_COLORS[c], alpha=0.65, s=50,
               edgecolors='white', lw=0.4,
               label=f'Cluster {c}  (fatality={sub["fat_rate"].mean()*100:.1f}%,  n={len(sub)})')
ax.set_title('City Risk Clustering — K-Means (k=4) PCA View', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)')
ax.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Cluster profiles table
st.markdown("#### Cluster Profiles")
summary = (ml.groupby('cluster_label')
           [['total_injured', 'total_dead', 'fat_rate', 'outside_city_inj']]
           .mean().round(2))
summary.index = [f'Cluster {i}' for i in summary.index]
summary.columns = ['Avg Injured', 'Avg Deaths', 'Fatality Rate', 'Avg Outside Injured']
summary['Fatality Rate'] = (summary['Fatality Rate'] * 100).round(1).astype(str) + '%'
st.dataframe(summary, use_container_width=True)

st.markdown("""<div class="viz-insight">
<strong>💡 Interpretation:</strong><br>
• <strong>Cluster 0 🔴:</strong> High fatality rate, low injury count → Deploy emergency units on highways<br>
• <strong>Cluster 1 🟠:</strong> High death volumes in major cities → Awareness campaigns + increased patrols during peak months<br>
• <strong>Cluster 2 🔵:</strong> Lowest risk → Routine monitoring sufficient<br>
• <strong>Cluster 3 🟢:</strong> Moderate risk → Improve road signage on major routes
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — KEY FINDINGS SUMMARY
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">📝 Key Findings Summary</div>', unsafe_allow_html=True)

findings = {
    "📉 Declining Trend": "Injuries decreased gradually from 1437 to 1439, indicating improved road safety.",
    "👨 Male Dominance": "Males represent ~83% of injuries and ~87% of deaths.",
    "🧑 Youth at Risk": "18–30 age group is the most affected in all cities.",
    "🏙️ Riyadh Leads": "Riyadh records the highest overall injury counts.",
    "🛣️ Highway Danger": "Outside-city accidents show greater severity and variability.",
    "📅 Ramadan Peak": "Month 9 consistently records the highest accident counts.",
    "⚠️ Fatal Roads": "Al-Baha and Tabuk have above-median fatality rates.",
    "🌍 Nationality": "Makkah, Madinah, Jeddah have high non-Saudi accident rates.",
    "📊 ML Models": f"Best model ({best}) achieved AUC = {results[best]['auc']:.3f}.",
    "🔬 4 Risk Clusters": "K-Means revealed 4 distinct city risk profiles.",
}

cols = st.columns(2)
for i, (title, desc) in enumerate(findings.items()):
    with cols[i % 2]:
        st.markdown(f"**{title}**")
        st.caption(desc)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #adb5bd; padding: 1rem 0 2rem;">
    Built with Streamlit &nbsp;•&nbsp; Tuwaiq Data Science & AI Bootcamp &nbsp;•&nbsp;
    Saudi Arabia Traffic Accidents Dataset (1437–1439 H)
</div>
""", unsafe_allow_html=True)
