"""
App.py — Streamlit Dashboard for Traffic Accidents EDA Project
==============================================================
This dashboard displays analysis results from the EDA project on
traffic accident data in Saudi Arabia (Hijri years 1437–1439).

It replicates the data pipeline from EDA_Project_Code.ipynb
without modifying the original notebook.

Run:  streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saudi Arabia Traffic Accidents Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Data Loading & Cleaning  (cached for performance)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """
    Replicate the data loading and cleaning steps from the
    EDA notebook (EDA_Project_Code.ipynb).

    Steps:
      1. Read three CSV files (semicolon-separated, utf-8-sig encoding)
      2. Concatenate into a single DataFrame
      3. Drop rows with all NaN values
      4. Rename Arabic columns to English
      5. Translate Arabic city names to English
      6. Create derived columns: total_accidents, binary injuries/deaths
    """
    # Determine the directory where App.py lives
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Step 1 & 2: Load and concatenate ---
    csv_files = [
        "Injured_and_Dead_in_Accidents_1437.csv",
        "Injured_and_Dead_in_Accidents_1438.csv",
        "Injured_and_Dead_in_Accidents_1439.csv",
    ]
    frames = []
    for f in csv_files:
        path = os.path.join(base_dir, f)
        frames.append(pd.read_csv(path, sep=";", encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True)

    # --- Step 3: Drop null rows ---
    df.dropna(inplace=True)

    # --- Step 4: Rename Arabic columns to English ---
    df.columns = [
        "month",          # الشهر
        "year",           # سنة
        "city",           # المدينة
        "males",          # ذكور
        "females",        # أناث
        "inside_city",    # داخل المدينه
        "outside_city",   # خارج المدينه
        "age_under18",    # أقل من 18
        "age_18_30",      # من18 إلى 30
        "age_30_40",      # من 30 على 40
        "age_40_50",      # من40إلى50
        "age_50plus",     # من50فأكثر
        "saudi",          # سعودى
        "non_saudi",      # غير سعودى
        "injuries",       # المصابين
        "deaths",         # المتوفين
    ]

    # --- Step 5: Translate city names ---
    df["city"] = df["city"].replace({
        "الرياض":          "Riyadh",
        "جده":             "Jeddah",
        "المدينه المنوره": "Madinah",
        "الشرقيه":         "Al-Sharqiyah",
        "الحدود الشماليه": "Northern Borders",
        "تبوك":            "Tabuk",
        "الجوف":           "Al-Jouf",
        "حائل":            "Hail",
        "نجران":           "Najran",
        "القصيم":          "Al-Qassim",
        "الباحه":          "Al-Baha",
        "عسير":            "Asir",
        "جازان":           "Jazan",
        "الطائف":          "Taif",
        "العاصمه":         "Makkah",
        "القريات":         "Al-Qurayyat",
    })

    # --- Step 6: Derived columns ---
    df["total_accidents"] = df["males"] + df["females"]
    df["injuries"] = (df["injuries"] > 0).astype(int)
    df["deaths"]   = (df["deaths"]  > 0).astype(int)

    return df


# Load data once
df = load_and_clean_data()


# ──────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📋 Project Overview",
        "🔍 Data Preview",
        "📊 Descriptive Statistics",
        "📈 Visualizations",
        "💡 Key Insights",
    ],
)

# ──────────────────────────────────────────────────────────────
# Helper: consistent Seaborn theme for all plots
# ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ══════════════════════════════════════════════════════════════
if page == "📋 Project Overview":
    st.title("🚗 Traffic Accidents Analysis in Saudi Arabia (1437–1439)")

    st.markdown(
        """
        > This dashboard presents the results of an **Exploratory Data Analysis (EDA)**
        > on traffic accident data across **16 cities in Saudi Arabia** over three
        > Hijri years (**1437, 1438, 1439**).
        >
        > The dataset contains monthly records covering **gender, age groups,
        > nationality**, and **location** (inside/outside city) of those involved
        > in traffic accidents.  The goal is to **explore patterns, identify
        > high-risk cities**, and surface actionable insights for road safety.
        """
    )

    # Quick summary metrics
    st.subheader("Dataset at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Years Covered", f"{int(df['year'].nunique())}")
    col3.metric("Cities", f"{df['city'].nunique()}")
    col4.metric("Total Accidents", f"{int(df['total_accidents'].sum()):,}")

    st.divider()

    # Year-wise totals as a quick bar
    st.subheader("Total Accidents by Year")
    yearly = df.groupby("year")["total_accidents"].sum().reset_index()
    yearly["year"] = yearly["year"].astype(int).astype(str)
    st.bar_chart(yearly.set_index("year")["total_accidents"])


# ══════════════════════════════════════════════════════════════
# PAGE 2 — Data Preview
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Data Preview":
    st.title("🔍 Data Preview")

    # Filters
    st.subheader("Filters")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        years = sorted(df["year"].unique())
        selected_years = st.multiselect(
            "Select Year(s)", years, default=years
        )
    with filter_col2:
        cities = sorted(df["city"].unique())
        selected_cities = st.multiselect(
            "Select City/Cities", cities, default=cities
        )

    filtered = df[
        (df["year"].isin(selected_years)) & (df["city"].isin(selected_cities))
    ]

    # Shape info
    st.info(
        f"Showing **{len(filtered):,}** rows × **{filtered.shape[1]}** columns "
        f"(out of {len(df):,} total rows)"
    )

    # Interactive dataframe
    st.dataframe(filtered, use_container_width=True, height=400)

    st.divider()

    # Column info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        "Column": filtered.columns,
        "Non-Null Count": [filtered[c].count() for c in filtered.columns],
        "Dtype": [str(filtered[c].dtype) for c in filtered.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — Descriptive Statistics
# ══════════════════════════════════════════════════════════════
elif page == "📊 Descriptive Statistics":
    st.title("📊 Descriptive Statistics")

    st.subheader("Numerical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.divider()

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values in the cleaned dataset! ✅")
    else:
        st.dataframe(missing[missing > 0], use_container_width=True)

    st.divider()

    # Value counts for categorical / binary columns
    st.subheader("Value Counts")
    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        st.markdown("**Cities**")
        st.dataframe(
            df["city"].value_counts().reset_index().rename(
                columns={"city": "City", "count": "Records"}
            ),
            hide_index=True,
        )
    with val_col2:
        st.markdown("**Years**")
        year_vc = df["year"].value_counts().sort_index().reset_index()
        year_vc.columns = ["Year", "Records"]
        year_vc["Year"] = year_vc["Year"].astype(int)
        st.dataframe(year_vc, hide_index=True)
    with val_col3:
        st.markdown("**Injuries / Deaths (binary)**")
        st.write(
            pd.DataFrame({
                "injuries=1": [int((df["injuries"] == 1).sum())],
                "injuries=0": [int((df["injuries"] == 0).sum())],
                "deaths=1":   [int((df["deaths"]   == 1).sum())],
                "deaths=0":   [int((df["deaths"]   == 0).sum())],
            }).T.rename(columns={0: "Count"})
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Visualizations
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.title("📈 Visualizations")

    viz_choice = st.selectbox(
        "Select a visualization",
        [
            "Total Accidents by Year (Line Chart)",
            "Monthly Accidents Heatmap",
            "Accidents by City",
            "Gender Distribution (Males vs Females)",
            "Age Group Distribution",
            "Inside vs Outside City",
            "Saudi vs Non-Saudi",
            "Feature Correlation Heatmap",
        ],
    )

    # --- 1. Total Accidents by Year ---
    if viz_choice == "Total Accidents by Year (Line Chart)":
        yearly = df.groupby("year", as_index=False)["total_accidents"].sum()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            yearly["year"], yearly["total_accidents"],
            marker="o", linewidth=2.5, color="#2563EB",
        )
        for _, row in yearly.iterrows():
            ax.annotate(
                f'{int(row["total_accidents"]):,}',
                (row["year"], row["total_accidents"]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=11, fontweight="bold",
            )
        ax.set_title("Total Accidents by Hijri Year", fontsize=16, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Accidents")
        ax.set_xticks(yearly["year"])
        ax.set_xticklabels(yearly["year"].astype(int))
        plt.tight_layout()
        st.pyplot(fig)

    # --- 2. Monthly Heatmap ---
    elif viz_choice == "Monthly Accidents Heatmap":
        pivot = df.pivot_table(
            values="total_accidents", index="year", columns="month", aggfunc="sum"
        )
        pivot.index = pivot.index.astype(int)
        pivot.columns = pivot.columns.astype(int)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax)
        ax.set_title("Monthly Total Accidents Heatmap", fontsize=16, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        plt.tight_layout()
        st.pyplot(fig)

    # --- 3. Accidents by City ---
    elif viz_choice == "Accidents by City":
        city_totals = (
            df.groupby("city")["total_accidents"]
            .sum()
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        city_totals.plot.barh(ax=ax, color="#2563EB")
        ax.set_title("Total Accidents by City", fontsize=16, fontweight="bold")
        ax.set_xlabel("Total Accidents")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)

    # --- 4. Gender Distribution ---
    elif viz_choice == "Gender Distribution (Males vs Females)":
        gender = pd.DataFrame({
            "Gender": ["Males", "Females"],
            "Total": [df["males"].sum(), df["females"].sum()],
        })
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(gender["Gender"], gender["Total"], color=["#2563EB", "#EC4899"])
        for i, v in enumerate(gender["Total"]):
            ax.text(i, v + 200, f"{int(v):,}", ha="center", fontweight="bold")
        ax.set_title("Gender Distribution", fontsize=16, fontweight="bold")
        ax.set_ylabel("Total Involved")
        plt.tight_layout()
        st.pyplot(fig)

    # --- 5. Age Group Distribution ---
    elif viz_choice == "Age Group Distribution":
        age_cols = ["age_under18", "age_18_30", "age_30_40", "age_40_50", "age_50plus"]
        age_labels = ["Under 18", "18–30", "30–40", "40–50", "50+"]
        age_totals = [df[c].sum() for c in age_cols]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(age_labels, age_totals, color="#8B5CF6")
        for bar, val in zip(bars, age_totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{int(val):,}", ha="center", fontweight="bold",
            )
        ax.set_title("Accidents by Age Group", fontsize=16, fontweight="bold")
        ax.set_ylabel("Total Involved")
        plt.tight_layout()
        st.pyplot(fig)

    # --- 6. Inside vs Outside City ---
    elif viz_choice == "Inside vs Outside City":
        loc_data = pd.DataFrame({
            "Location": ["Inside City", "Outside City"],
            "Total": [df["inside_city"].sum(), df["outside_city"].sum()],
        })
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(loc_data["Location"], loc_data["Total"], color=["#10B981", "#F59E0B"])
        for i, v in enumerate(loc_data["Total"]):
            ax.text(i, v + 200, f"{int(v):,}", ha="center", fontweight="bold")
        ax.set_title("Inside City vs Outside City", fontsize=16, fontweight="bold")
        ax.set_ylabel("Total Involved")
        plt.tight_layout()
        st.pyplot(fig)

    # --- 7. Saudi vs Non-Saudi ---
    elif viz_choice == "Saudi vs Non-Saudi":
        nat_data = pd.DataFrame({
            "Nationality": ["Saudi", "Non-Saudi"],
            "Total": [df["saudi"].sum(), df["non_saudi"].sum()],
        })
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(nat_data["Nationality"], nat_data["Total"], color=["#059669", "#DC2626"])
        for i, v in enumerate(nat_data["Total"]):
            ax.text(i, v + 200, f"{int(v):,}", ha="center", fontweight="bold")
        ax.set_title("Saudi vs Non-Saudi Involvement", fontsize=16, fontweight="bold")
        ax.set_ylabel("Total Involved")
        plt.tight_layout()
        st.pyplot(fig)

    # --- 8. Correlation Heatmap ---
    elif viz_choice == "Feature Correlation Heatmap":
        num_cols = [
            "males", "females", "inside_city", "outside_city",
            "age_18_30", "age_30_40", "saudi", "non_saudi",
            "injuries", "deaths", "total_accidents",
        ]
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.6, square=True, annot_kws={"size": 9},
            cbar_kws={"label": "Correlation"}, ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — Key Insights
# ══════════════════════════════════════════════════════════════
elif page == "💡 Key Insights":
    st.title("💡 Key Insights & Findings")

    st.markdown(
        """
        The following insights were extracted from the exploratory data analysis:

        ---

        ### 📉 1. Declining Accident Trend
        Total accidents show a **consistent decline** across the three years:
        - **1437**: 47,151
        - **1438**: 40,688
        - **1439**: 36,242

        This represents a **~23% decrease** from 1437 to 1439.

        ---

        ### 👨 2. Gender Disparity
        **Males** represent the overwhelming majority of those involved in
        traffic accidents across all cities and years.

        ---

        ### 🧑 3. High-Risk Age Group
        The **18–30 age group** is the most frequently involved in accidents
        across all cities, followed by the 30–40 age group.

        ---

        ### 🏙️ 4. Riyadh Leads in Accidents
        **Riyadh** records the highest number of accidents by a significant
        margin, consistent across all three years.

        ---

        ### 🛣️ 5. Outside-City Accidents
        Accidents occurring **outside city limits** show higher spread and
        more extreme values, suggesting highway-related incidents are more severe.

        ---

        ### 📅 6. Seasonal Spike in Month 9
        **Month 9** consistently shows a spike in accidents across all three
        years, potentially linked to seasonal factors or holiday periods.

        ---

        ### 🤖 7. Predictive Model (Linear Regression)
        A simple linear regression model trained on monthly data achieved:
        - **R² = 0.77** — explains 77% of variance
        - **MAE = 150** — average prediction error of ~150 accidents

        The model forecasts a continued **downward trend** in accident numbers
        for the following months.
        """
    )


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit • Data from Saudi Arabia Traffic Accidents Dataset")
