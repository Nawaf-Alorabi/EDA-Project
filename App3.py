import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================
# إعدادات الصفحة
# ============================
st.set_page_config(
    page_title="تحليل حوادث المرور - المملكة العربية السعودية",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CSS للتصميم الاحترافي
# ============================
st.markdown("""
<style>
    /* الخلفية العامة */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* البطاقات */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    
    .metric-card h3 {
        font-size: 18px;
        margin-bottom: 10px;
        font-weight: 500;
        opacity: 0.9;
    }
    
    .metric-card h1 {
        font-size: 42px;
        margin: 10px 0;
        font-weight: 700;
    }
    
    /* العنوان الرئيسي */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .main-header p {
        font-size: 18px;
        opacity: 0.9;
    }
    
    /* الأقسام */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 30px 0 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .section-header h2 {
        margin: 0;
        font-size: 26px;
        font-weight: 600;
    }
    
    /* الشريط الجانبي */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] h1, h2, h3, p {
        color: white !important;
    }
    
    /* إخفاء عناصر Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* مربعات المعلومات */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# تحميل البيانات الحقيقية
# ============================
@st.cache_data
def load_real_data():
    """تحميل البيانات الحقيقية من الملفات"""
    
    CITIES = ['الطائف', 'تبوك', 'القصيم', 'حائل', 'الباحة', 'جازان', 'نجران', 'الجوف']
    
    MONTHS_AR = ['محرم', 'صفر', 'ربيع أول', 'ربيع ثانى', 'جمادى أول', 'جمادى ثانى',
                 'رجب', 'شعبان', 'رمضان', 'شوال', 'ذى القعدة', 'ذى الحجة']
    
    def clean_value(val):
        if pd.isna(val) or val == '':
            return 0
        try:
            return int(float(str(val).replace(',', '')))
        except:
            return 0
    
    all_data = []
    
    for year in [1437, 1438, 1439]:
        file = f'Injured_in_Accidents_{year}_H.csv'
        
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            
            for city in CITIES:
                city_mask = df.iloc[:, 0].astype(str).str.contains(city, na=False)
                
                if city_mask.any():
                    city_start_idx = df[city_mask].index[0]
                    city_block = df.iloc[city_start_idx:city_start_idx+15]
                    month_data = city_block.iloc[2:14]
                    
                    for idx, row in enumerate(month_data.itertuples()):
                        if idx < 12:
                            month_name = str(row[1]).strip() if len(row) > 0 else ''
                            month_num = idx + 1
                            
                            for m_idx, m_name in enumerate(MONTHS_AR):
                                if m_name in month_name:
                                    month_num = m_idx + 1
                                    break
                            
                            male = clean_value(row[2] if len(row) > 2 else 0)
                            female = clean_value(row[3] if len(row) > 3 else 0)
                            in_city = clean_value(row[5] if len(row) > 5 else 0)
                            out_city = clean_value(row[6] if len(row) > 6 else 0)
                            saudi = clean_value(row[14] if len(row) > 14 else 0)
                            non_saudi = clean_value(row[15] if len(row) > 15 else 0)
                            
                            all_data.append({
                                'city': city,
                                'year': year,
                                'month': month_num,
                                'month_name': month_name,
                                'male_injured': male,
                                'female_injured': female,
                                'total_injured': male + female,
                                'in_city': in_city,
                                'out_city': out_city,
                                'saudi': saudi,
                                'non_saudi': non_saudi,
                                'male_dead': 0,
                                'female_dead': 0
                            })
        except:
            continue
    
    if all_data:
        df_result = pd.DataFrame(all_data)
        df_result['total_cases'] = df_result['total_injured']
        df_result['total_dead'] = df_result['male_dead'] + df_result['female_dead']
        return df_result
    else:
        return None

# تحميل البيانات
df = load_real_data()

if df is None:
    st.error("❌ لم يتم العثور على ملفات البيانات. يرجى التأكد من وجود الملفات في المجلد.")
    st.stop()

# ============================
# العنوان الرئيسي
# ============================
st.markdown("""
<div class="main-header">
    <h1>🚦 تحليل حوادث المرور في المملكة العربية السعودية</h1>
    <p>الإدارة العامة للمرور - وزارة الداخلية | البيانات الحقيقية: ١٤٣٧-١٤٣٩ هـ</p>
</div>
""", unsafe_allow_html=True)

# ============================
# الشريط الجانبي - الفلاتر
# ============================
st.sidebar.markdown("## 🎛️ لوحة التحكم")
st.sidebar.markdown("---")

selected_years = st.sidebar.multiselect(
    "اختر السنة (هـ)",
    options=sorted(df['year'].unique()),
    default=sorted(df['year'].unique())
)

selected_cities = st.sidebar.multiselect(
    "اختر المدينة",
    options=sorted(df['city'].unique()),
    default=sorted(df['city'].unique())
)

selected_months = st.sidebar.slider(
    "اختر نطاق الأشهر",
    min_value=1,
    max_value=12,
    value=(1, 12)
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; color: white;'>
    <h4 style='color: white;'>📊 البيانات</h4>
    <p style='font-size: 14px;'>المدن المتاحة: {len(df['city'].unique())}</p>
    <p style='font-size: 14px;'>إجمالي السجلات: {len(df):,}</p>
</div>
""", unsafe_allow_html=True)

filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['city'].isin(selected_cities)) &
    (df['month'] >= selected_months[0]) &
    (df['month'] <= selected_months[1])
]

if len(filtered_df) == 0:
    st.warning("⚠️ لا توجد بيانات مطابقة للفلاتر المحددة. يرجى تعديل الفلاتر.")
    st.stop()

# ============================
# المقاييس الرئيسية
# ============================
total_injured = filtered_df['total_injured'].sum()
male_injured = filtered_df['male_injured'].sum()
female_injured = filtered_df['female_injured'].sum()
male_percentage = (male_injured / total_injured * 100) if total_injured > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>إجمالي المصابين</h3>
        <h1>{total_injured:,}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>ذكور مصابون</h3>
        <h1>{male_injured:,}</h1>
        <p style='font-size: 14px; margin-top: 5px;'>{male_percentage:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>إناث مصابات</h3>
        <h1>{female_injured:,}</h1>
        <p style='font-size: 14px; margin-top: 5px;'>{100-male_percentage:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_monthly = total_injured / len(filtered_df) if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>متوسط شهري</h3>
        <h1>{avg_monthly:.0f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ============================
# القسم 1: الاتجاه الزمني
# ============================
st.markdown("""
<div class="section-header">
    <h2>📈 الاتجاه الزمني للحوادث</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    yearly_data = filtered_df.groupby('year').agg({
        'total_injured': 'sum',
        'male_injured': 'sum',
        'female_injured': 'sum'
    }).reset_index()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=yearly_data['year'],
        y=yearly_data['total_injured'],
        name='إجمالي المصابين',
        marker_color='#667eea',
        text=yearly_data['total_injured'],
        textposition='outside',
        texttemplate='%{text:,.0f}'
    ))
    
    fig1.update_layout(
        title='إجمالي المصابين حسب السنة',
        xaxis_title='السنة الهجرية',
        yaxis_title='عدد المصابين',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    if len(yearly_data) > 1:
        change_pct = ((yearly_data.iloc[-1]['total_injured'] - yearly_data.iloc[0]['total_injured']) / 
                      yearly_data.iloc[0]['total_injured'] * 100)
        change_color = 'green' if change_pct < 0 else 'red'
        change_symbol = '↓' if change_pct < 0 else '↑'
        st.markdown(f"""
        <div style='background: white; padding: 15px; border-radius: 10px; text-align: center;'>
            <p style='color: {change_color}; font-size: 20px; font-weight: bold; margin: 0;'>
                {change_symbol} {abs(change_pct):.1f}%
            </p>
            <p style='color: #666; font-size: 14px; margin: 5px 0 0 0;'>
                {'انخفاض' if change_pct < 0 else 'ارتفاع'} من {yearly_data.iloc[0]['year']} إلى {yearly_data.iloc[-1]['year']}
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    monthly_data = filtered_df.groupby(['year', 'month'])['total_injured'].sum().reset_index()
    
    fig2 = go.Figure()
    
    colors = {1437: '#2980B9', 1438: '#1ABC9C', 1439: '#8E44AD'}
    
    for year in sorted(monthly_data['year'].unique()):
        year_data = monthly_data[monthly_data['year'] == year]
        fig2.add_trace(go.Scatter(
            x=year_data['month'],
            y=year_data['total_injured'],
            name=f'{year} هـ',
            mode='lines+markers',
            line=dict(width=3, color=colors.get(year, '#E74C3C')),
            marker=dict(size=8)
        ))
    
    fig2.add_vrect(
        x0=8.5, x1=9.5,
        fillcolor="orange", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="رمضان",
        annotation_position="top"
    )
    
    fig2.update_layout(
        title='الاتجاه الشهري للمصابين عبر السنوات',
        xaxis_title='الشهر',
        yaxis_title='عدد المصابين',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================
# القسم 2: التحليل الجغرافي
# ============================
st.markdown("""
<div class="section-header">
    <h2>🗺️ التوزيع الجغرافي للمصابين</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    city_data = filtered_df.groupby('city').agg({
        'total_injured': 'sum',
        'male_injured': 'sum',
        'female_injured': 'sum'
    }).reset_index()
    city_data = city_data.sort_values('total_injured', ascending=True)
    
    fig3 = go.Figure(go.Bar(
        y=city_data['city'],
        x=city_data['total_injured'],
        orientation='h',
        marker=dict(
            color=city_data['total_injured'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="عدد المصابين")
        ),
        text=city_data['total_injured'],
        textposition='outside',
        texttemplate='%{text:,.0f}'
    ))
    
    fig3.update_layout(
        title='المدن حسب عدد المصابين',
        xaxis_title='عدد المصابين',
        yaxis_title='المدينة',
        template='plotly_dark',
        height=450
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    city_gender = filtered_df.groupby('city').agg({
        'male_injured': 'sum',
        'female_injured': 'sum',
        'total_injured': 'sum'
    }).reset_index()
    city_gender['male_pct'] = (city_gender['male_injured'] / city_gender['total_injured'] * 100).round(1)
    city_gender = city_gender.sort_values('total_injured', ascending=True)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=city_gender['city'],
        x=city_gender['male_injured'],
        name='ذكور',
        orientation='h',
        marker_color='#3498db',
        text=city_gender['male_injured'],
        textposition='inside'
    ))
    fig4.add_trace(go.Bar(
        y=city_gender['city'],
        x=city_gender['female_injured'],
        name='إناث',
        orientation='h',
        marker_color='#e74c3c',
        text=city_gender['female_injured'],
        textposition='inside'
    ))
    
    fig4.update_layout(
        title='توزيع المصابين حسب الجنس والمدينة',
        xaxis_title='عدد المصابين',
        yaxis_title='المدينة',
        template='plotly_dark',
        height=450,
        barmode='stack'
    )
    st.plotly_chart(fig4, use_container_width=True)

# ============================
# القسم 3: التحليل الديموغرافي
# ============================
st.markdown("""
<div class="section-header">
    <h2>👥 التحليل الديموغرافي</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender_data = pd.DataFrame({
        'الفئة': ['ذكور', 'إناث'],
        'العدد': [
            filtered_df['male_injured'].sum(),
            filtered_df['female_injured'].sum()
        ]
    })
    
    fig5 = go.Figure(data=[go.Pie(
        labels=gender_data['الفئة'],
        values=gender_data['العدد'],
        hole=0.4,
        marker=dict(colors=['#3498db', '#e74c3c']),
        textinfo='label+percent+value',
        textposition='outside',
        texttemplate='%{label}<br>%{value:,}<br>%{percent}'
    )])
    
    fig5.update_layout(
        title='توزيع المصابين حسب الجنس',
        template='plotly_dark',
        height=350
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    location_data = pd.DataFrame({
        'الموقع': ['داخل المدينة', 'خارج المدينة'],
        'العدد': [
            filtered_df['in_city'].sum(),
            filtered_df['out_city'].sum()
        ]
    })
    
    fig6 = go.Figure(data=[go.Pie(
        labels=location_data['الموقع'],
        values=location_data['العدد'],
        hole=0.4,
        marker=dict(colors=['#16a085', '#d35400']),
        textinfo='label+percent+value',
        textposition='outside',
        texttemplate='%{label}<br>%{value:,}<br>%{percent}'
    )])
    
    fig6.update_layout(
        title='توزيع الحوادث حسب الموقع',
        template='plotly_dark',
        height=350
    )
    st.plotly_chart(fig6, use_container_width=True)

with col3:
    nationality_data = pd.DataFrame({
        'الجنسية': ['سعودي', 'غير سعودي'],
        'العدد': [
            filtered_df['saudi'].sum(),
            filtered_df['non_saudi'].sum()
        ]
    })
    
    fig7 = go.Figure(data=[go.Pie(
        labels=nationality_data['الجنسية'],
        values=nationality_data['العدد'],
        hole=0.4,
        marker=dict(colors=['#27ae60', '#c0392b']),
        textinfo='label+percent+value',
        textposition='outside',
        texttemplate='%{label}<br>%{value:,}<br>%{percent}'
    )])
    
    fig7.update_layout(
        title='توزيع المصابين حسب الجنسية',
        template='plotly_dark',
        height=350
    )
    st.plotly_chart(fig7, use_container_width=True)

# ============================
# القسم 4: تحليل الأنماط الشهرية
# ============================
st.markdown("""
<div class="section-header">
    <h2>📅 تحليل الأنماط الشهرية</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    monthly_avg = filtered_df.groupby('month')['total_injured'].mean().reset_index()
    monthly_avg['month_name'] = ['محرم', 'صفر', 'ر.أول', 'ر.ثاني', 'ج.أول', 'ج.ثاني', 
                                  'رجب', 'شعبان', 'رمضان', 'شوال', 'ذو القعدة', 'ذو الحجة']
    
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg['total_injured'],
        marker=dict(
            color=monthly_avg['total_injured'],
            colorscale='Viridis',
            showscale=True
        ),
        text=monthly_avg['total_injured'].round(0),
        textposition='outside',
        texttemplate='%{text:.0f}'
    ))
    
    fig8.update_layout(
        title='متوسط المصابين لكل شهر',
        xaxis_title='الشهر',
        yaxis_title='متوسط عدد المصابين',
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig8, use_container_width=True)

with col2:
    monthly_location = filtered_df.groupby('month').agg({
        'in_city': 'sum',
        'out_city': 'sum'
    }).reset_index()
    monthly_location['month_name'] = ['محرم', 'صفر', 'ر.أول', 'ر.ثاني', 'ج.أول', 'ج.ثاني', 
                                       'رجب', 'شعبان', 'رمضان', 'شوال', 'ذو القعدة', 'ذو الحجة']
    
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(
        x=monthly_location['month_name'],
        y=monthly_location['in_city'],
        name='داخل المدينة',
        mode='lines+markers',
        line=dict(width=3, color='#16a085'),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(22, 160, 133, 0.3)'
    ))
    fig9.add_trace(go.Scatter(
        x=monthly_location['month_name'],
        y=monthly_location['out_city'],
        name='خارج المدينة',
        mode='lines+markers',
        line=dict(width=3, color='#d35400'),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(211, 84, 0, 0.3)'
    ))
    
    fig9.update_layout(
        title='توزيع المصابين حسب الموقع عبر الأشهر',
        xaxis_title='الشهر',
        yaxis_title='عدد المصابين',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig9, use_container_width=True)

# ============================
# القسم 5: Heatmap
# ============================
st.markdown("""
<div class="section-header">
    <h2>🔥 خريطة الحرارة للمصابين</h2>
</div>
""", unsafe_allow_html=True)

heatmap_data = filtered_df.groupby(['city', 'month'])['total_injured'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='city', columns='month', values='total_injured').fillna(0)

month_names = ['محرم', 'صفر', 'ر.أول', 'ر.ثاني', 'ج.أول', 'ج.ثاني', 
               'رجب', 'شعبان', 'رمضان', 'شوال', 'ذو القعدة', 'ذو الحجة']

fig10 = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[month_names[i-1] for i in heatmap_pivot.columns],
    y=heatmap_pivot.index,
    colorscale='YlOrRd',
    text=heatmap_pivot.values,
    texttemplate='%{text:.0f}',
    textfont={"size": 11},
    colorbar=dict(title="عدد المصابين"),
    hovertemplate='المدينة: %{y}<br>الشهر: %{x}<br>المصابين: %{z}<extra></extra>'
))

fig10.update_layout(
    title='خريطة الحرارة: توزيع المصابين حسب المدينة والشهر',
    xaxis_title='الشهر',
    yaxis_title='المدينة',
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig10, use_container_width=True)

# ============================
# القسم 6: الاستنتاجات
# ============================
st.markdown("""
<div class="section-header">
    <h2>💡 الاستنتاجات والتوصيات الرئيسية</h2>
</div>
""", unsafe_allow_html=True)

total_1437 = df[df['year'] == 1437]['total_injured'].sum()
total_1439 = df[df['year'] == 1439]['total_injured'].sum()
decrease_pct = ((total_1437 - total_1439) / total_1437 * 100)
top_city = df.groupby('city')['total_injured'].sum().idxmax()
ramadan_avg = df[df['month'] == 9]['total_injured'].mean()
other_months_avg = df[df['month'] != 9]['total_injured'].mean()
ramadan_increase = ((ramadan_avg - other_months_avg) / other_months_avg * 100)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="info-box">
        <h3 style='color: #2c3e50;'>📊 الاستنتاجات الرئيسية:</h3>
        <ul style='color: #34495e; line-height: 1.8;'>
            <li><strong>انخفاض تدريجي:</strong> انخفاض بنسبة {decrease_pct:.1f}% من ١٤٣٧ إلى ١٤٣٩ هـ ({total_1437:,} → {total_1439:,})</li>
            <li><strong>شهر رمضان:</strong> يسجل زيادة {ramadan_increase:.1f}% عن باقي الأشهر</li>
            <li><strong>الذكور:</strong> يشكلون {male_percentage:.1f}% من إجمالي المصابين</li>
            <li><strong>أعلى مدينة:</strong> {top_city} بإجمالي {df[df['city']==top_city]['total_injured'].sum():,} مصاب</li>
            <li><strong>خارج المدن:</strong> {(filtered_df['out_city'].sum() / (filtered_df['in_city'].sum() + filtered_df['out_city'].sum()) * 100):.1f}% من الحوادث</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <h3 style='color: #2c3e50;'>🎯 التوصيات الاستراتيجية:</h3>
        <ul style='color: #34495e; line-height: 1.8;'>
            <li><strong>حملات توعوية:</strong> تكثيف الحملات في شهر رمضان</li>
            <li><strong>الطرق السريعة:</strong> تحسين الإجراءات الأمنية خارج المدن</li>
            <li><strong>الشباب:</strong> برامج توعية مستهدفة للذكور</li>
            <li><strong>التكنولوجيا:</strong> استخدام الذكاء الاصطناعي للتنبؤ</li>
            <li><strong>المراقبة:</strong> زيادة الرقابة في المناطق عالية الخطورة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================
# القسم 7: جدول البيانات
# ============================
st.markdown("""
<div class="section-header">
    <h2>📋 جدول البيانات التفصيلي</h2>
</div>
""", unsafe_allow_html=True)

display_df = filtered_df.groupby(['year', 'city']).agg({
    'total_injured': 'sum',
    'male_injured': 'sum',
    'female_injured': 'sum',
    'in_city': 'sum',
    'out_city': 'sum'
}).reset_index()

display_df = display_df.sort_values('total_injured', ascending=False)
display_df.columns = ['السنة', 'المدينة', 'إجمالي المصابين', 'ذكور', 'إناث', 'داخل المدينة', 'خارج المدينة']

st.dataframe(
    display_df.style.background_gradient(cmap='RdYlGn_r', subset=['إجمالي المصابين']),
    use_container_width=True,
    height=400
)

csv = display_df.to_csv(index=False, encoding='utf-8-sig')
st.download_button(
    label="📥 تحميل البيانات (CSV)",
    data=csv,
    file_name=f'traffic_accidents_data_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
    mime='text/csv'
)

# ============================
# التذييل
# ============================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: white; padding: 20px;'>
    <p style='font-size: 16px;'>© ٢٠٢٦ الإدارة العامة للمرور - وزارة الداخلية | المملكة العربية السعودية</p>
    <p style='font-size: 14px; opacity: 0.8;'>تم التطوير باستخدام Streamlit و Plotly | البيانات الحقيقية: {len(df):,} سجل</p>
    <p style='font-size: 12px; opacity: 0.7;'>المدن المتاحة: {', '.join(sorted(df['city'].unique()))}</p>
</div>
""", unsafe_allow_html=True)
