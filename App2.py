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
# إنشاء البيانات التجريبية
# ============================
@st.cache_data
def generate_data():
    """إنشاء بيانات تجريبية لحوادث المرور"""
    
    cities = ['الرياض', 'جدة', 'مكة المكرمة', 'المدينة المنورة', 'الدمام', 'الأحساء', 
              'الطائف', 'تبوك', 'القصيم', 'حائل', 'الباحة', 'جازان', 'نجران', 
              'الجوف', 'عرعر', 'ينبع']
    
    years = [1437, 1438, 1439]
    months = list(range(1, 13))
    
    np.random.seed(42)
    
    data_list = []
    for city in cities:
        for year in years:
            for month in months:
                # نمط انخفاض تدريجي عبر السنوات
                base = 150 if city in ['الرياض', 'جدة', 'مكة المكرمة'] else 80
                year_factor = 1.0 - (year - 1437) * 0.15  # انخفاض 15% سنوياً
                
                # ذروة في رمضان (شهر 9)
                ramadan_boost = 1.4 if month == 9 else 1.0
                
                # تباين طبيعي
                noise = np.random.uniform(0.85, 1.15)
                
                male_injured = int(base * year_factor * ramadan_boost * noise * 0.83)
                female_injured = int(base * year_factor * ramadan_boost * noise * 0.17)
                
                male_dead = int(male_injured * 0.08 * np.random.uniform(0.8, 1.2))
                female_dead = int(female_injured * 0.06 * np.random.uniform(0.8, 1.2))
                
                # حوادث داخل وخارج المدينة
                in_city = int((male_injured + female_injured) * np.random.uniform(0.65, 0.75))
                out_city = (male_injured + female_injured) - in_city
                
                # الجنسية
                saudi = int((male_injured + female_injured) * np.random.uniform(0.7, 0.9))
                non_saudi = (male_injured + female_injured) - saudi
                
                data_list.append({
                    'city': city,
                    'year': year,
                    'month': month,
                    'male_injured': male_injured,
                    'female_injured': female_injured,
                    'male_dead': male_dead,
                    'female_dead': female_dead,
                    'in_city': in_city,
                    'out_city': out_city,
                    'saudi': saudi,
                    'non_saudi': non_saudi
                })
    
    df = pd.DataFrame(data_list)
    
    # حساب الأعمدة الإضافية
    df['total_injured'] = df['male_injured'] + df['female_injured']
    df['total_dead'] = df['male_dead'] + df['female_dead']
    df['total_cases'] = df['total_injured'] + df['total_dead']
    df['fatality_rate'] = (df['total_dead'] / df['total_cases'] * 100).round(2)
    
    return df

# تحميل البيانات
df = generate_data()

# ============================
# العنوان الرئيسي
# ============================
st.markdown("""
<div class="main-header">
    <h1>🚦 تحليل حوادث المرور في المملكة العربية السعودية</h1>
    <p>الإدارة العامة للمرور - وزارة الداخلية | البيانات: ١٤٣٧-١٤٣٩ هـ</p>
</div>
""", unsafe_allow_html=True)

# ============================
# الشريط الجانبي - الفلاتر
# ============================
st.sidebar.markdown("## 🎛️ لوحة التحكم")
st.sidebar.markdown("---")

# فلتر السنة
selected_years = st.sidebar.multiselect(
    "اختر السنة (هـ)",
    options=sorted(df['year'].unique()),
    default=sorted(df['year'].unique())
)

# فلتر المدينة
selected_cities = st.sidebar.multiselect(
    "اختر المدينة",
    options=sorted(df['city'].unique()),
    default=sorted(df['city'].unique())[:5]
)

# فلتر الشهر
selected_months = st.sidebar.slider(
    "اختر نطاق الأشهر",
    min_value=1,
    max_value=12,
    value=(1, 12)
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; color: white;'>
    <h4 style='color: white;'>💡 نصيحة</h4>
    <p style='font-size: 14px;'>استخدم الفلاتر أعلاه لتخصيص التحليل حسب احتياجاتك</p>
</div>
""", unsafe_allow_html=True)

# تطبيق الفلاتر
filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['city'].isin(selected_cities)) &
    (df['month'] >= selected_months[0]) &
    (df['month'] <= selected_months[1])
]

# ============================
# المقاييس الرئيسية
# ============================
total_injured = filtered_df['total_injured'].sum()
total_dead = filtered_df['total_dead'].sum()
total_cases = filtered_df['total_cases'].sum()
avg_fatality = filtered_df['fatality_rate'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>إجمالي الحوادث</h3>
        <h1>{total_cases:,}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>إجمالي المصابين</h3>
        <h1>{total_injured:,}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>إجمالي الوفيات</h3>
        <h1>{total_dead:,}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>معدل الوفيات</h3>
        <h1>{avg_fatality:.1f}%</h1>
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
    # الاتجاه السنوي
    yearly_data = filtered_df.groupby('year').agg({
        'total_cases': 'sum',
        'total_injured': 'sum',
        'total_dead': 'sum'
    }).reset_index()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=yearly_data['year'],
        y=yearly_data['total_cases'],
        name='إجمالي الحوادث',
        marker_color='#667eea',
        text=yearly_data['total_cases'],
        textposition='outside',
        texttemplate='%{text:,.0f}'
    ))
    
    fig1.update_layout(
        title='إجمالي الحوادث حسب السنة',
        xaxis_title='السنة الهجرية',
        yaxis_title='عدد الحوادث',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # الاتجاه الشهري
    monthly_data = filtered_df.groupby(['year', 'month'])['total_cases'].sum().reset_index()
    
    fig2 = go.Figure()
    
    colors = {'1437': '#2980B9', '1438': '#1ABC9C', '1439': '#8E44AD'}
    
    for year in sorted(monthly_data['year'].unique()):
        year_data = monthly_data[monthly_data['year'] == year]
        fig2.add_trace(go.Scatter(
            x=year_data['month'],
            y=year_data['total_cases'],
            name=f'{year} هـ',
            mode='lines+markers',
            line=dict(width=3, color=colors.get(str(year), '#E74C3C')),
            marker=dict(size=8)
        ))
    
    # تظليل رمضان
    fig2.add_vrect(
        x0=8.5, x1=9.5,
        fillcolor="orange", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="رمضان",
        annotation_position="top"
    )
    
    fig2.update_layout(
        title='الاتجاه الشهري للحوادث عبر السنوات',
        xaxis_title='الشهر',
        yaxis_title='عدد الحوادث',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================
# القسم 2: التحليل الجغرافي
# ============================
st.markdown("""
<div class="section-header">
    <h2>🗺️ التوزيع الجغرافي للحوادث</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # أعلى المدن في الحوادث
    city_data = filtered_df.groupby('city').agg({
        'total_cases': 'sum',
        'total_injured': 'sum',
        'total_dead': 'sum',
        'fatality_rate': 'mean'
    }).reset_index()
    city_data = city_data.sort_values('total_cases', ascending=False).head(10)
    
    fig3 = go.Figure(go.Bar(
        y=city_data['city'],
        x=city_data['total_cases'],
        orientation='h',
        marker=dict(
            color=city_data['total_cases'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="عدد الحوادث")
        ),
        text=city_data['total_cases'],
        textposition='outside',
        texttemplate='%{text:,.0f}'
    ))
    
    fig3.update_layout(
        title='أعلى 10 مدن في عدد الحوادث',
        xaxis_title='عدد الحوادث',
        yaxis_title='المدينة',
        template='plotly_dark',
        height=450,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # معدل الوفيات حسب المدينة
    city_fatality = filtered_df.groupby('city')['fatality_rate'].mean().reset_index()
    city_fatality = city_fatality.sort_values('fatality_rate', ascending=False).head(10)
    
    fig4 = go.Figure(go.Bar(
        y=city_fatality['city'],
        x=city_fatality['fatality_rate'],
        orientation='h',
        marker=dict(
            color=city_fatality['fatality_rate'],
            colorscale='Oranges',
            showscale=True,
            colorbar=dict(title="معدل الوفيات %")
        ),
        text=city_fatality['fatality_rate'].round(1),
        textposition='outside',
        texttemplate='%{text}%'
    ))
    
    fig4.update_layout(
        title='أعلى 10 مدن في معدل الوفيات',
        xaxis_title='معدل الوفيات (%)',
        yaxis_title='المدينة',
        template='plotly_dark',
        height=450,
        yaxis=dict(autorange="reversed")
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
    # توزيع الجنس
    gender_data = pd.DataFrame({
        'الفئة': ['ذكور مصابون', 'إناث مصابات', 'ذكور متوفون', 'إناث متوفيات'],
        'العدد': [
            filtered_df['male_injured'].sum(),
            filtered_df['female_injured'].sum(),
            filtered_df['male_dead'].sum(),
            filtered_df['female_dead'].sum()
        ]
    })
    
    fig5 = go.Figure(data=[go.Pie(
        labels=gender_data['الفئة'],
        values=gender_data['العدد'],
        hole=0.4,
        marker=dict(colors=['#3498db', '#e74c3c', '#2c3e50', '#95a5a6']),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig5.update_layout(
        title='توزيع الحوادث حسب الجنس',
        template='plotly_dark',
        height=350
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    # موقع الحادث
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
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig6.update_layout(
        title='توزيع الحوادث حسب الموقع',
        template='plotly_dark',
        height=350
    )
    st.plotly_chart(fig6, use_container_width=True)

with col3:
    # الجنسية
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
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig7.update_layout(
        title='توزيع الحوادث حسب الجنسية',
        template='plotly_dark',
        height=350
    )
    st.plotly_chart(fig7, use_container_width=True)

# ============================
# القسم 4: مقارنة الإصابات والوفيات
# ============================
st.markdown("""
<div class="section-header">
    <h2>⚖️ مقارنة الإصابات والوفيات</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # مقارنة الإصابات والوفيات عبر السنوات
    comparison_data = filtered_df.groupby('year').agg({
        'total_injured': 'sum',
        'total_dead': 'sum'
    }).reset_index()
    
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(
        name='المصابون',
        x=comparison_data['year'],
        y=comparison_data['total_injured'],
        marker_color='#3498db',
        text=comparison_data['total_injured'],
        textposition='outside'
    ))
    fig8.add_trace(go.Bar(
        name='المتوفون',
        x=comparison_data['year'],
        y=comparison_data['total_dead'],
        marker_color='#e74c3c',
        text=comparison_data['total_dead'],
        textposition='outside'
    ))
    
    fig8.update_layout(
        title='مقارنة الإصابات والوفيات عبر السنوات',
        xaxis_title='السنة',
        yaxis_title='العدد',
        template='plotly_dark',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig8, use_container_width=True)

with col2:
    # نسبة الوفيات عبر الأشهر
    monthly_fatality = filtered_df.groupby('month').agg({
        'total_dead': 'sum',
        'total_cases': 'sum'
    }).reset_index()
    monthly_fatality['fatality_rate'] = (monthly_fatality['total_dead'] / monthly_fatality['total_cases'] * 100).round(2)
    
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(
        x=monthly_fatality['month'],
        y=monthly_fatality['fatality_rate'],
        mode='lines+markers',
        line=dict(width=3, color='#e74c3c'),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig9.update_layout(
        title='معدل الوفيات عبر الأشهر',
        xaxis_title='الشهر',
        yaxis_title='معدل الوفيات (%)',
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig9, use_container_width=True)

# ============================
# القسم 5: Heatmap الحوادث
# ============================
st.markdown("""
<div class="section-header">
    <h2>🔥 خريطة الحرارة للحوادث</h2>
</div>
""", unsafe_allow_html=True)

# Heatmap للمدن والأشهر
heatmap_data = filtered_df.groupby(['city', 'month'])['total_cases'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='city', columns='month', values='total_cases').fillna(0)

# اختيار أعلى 12 مدينة
top_cities = filtered_df.groupby('city')['total_cases'].sum().nlargest(12).index
heatmap_pivot = heatmap_pivot.loc[top_cities]

fig10 = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[f'شهر {i}' for i in heatmap_pivot.columns],
    y=heatmap_pivot.index,
    colorscale='Reds',
    text=heatmap_pivot.values,
    texttemplate='%{text:.0f}',
    textfont={"size": 10},
    colorbar=dict(title="عدد الحوادث")
))

fig10.update_layout(
    title='خريطة الحرارة: توزيع الحوادث حسب المدينة والشهر',
    xaxis_title='الشهر',
    yaxis_title='المدينة',
    template='plotly_dark',
    height=500
)
st.plotly_chart(fig10, use_container_width=True)

# ============================
# القسم 6: الاستنتاجات الرئيسية
# ============================
st.markdown("""
<div class="section-header">
    <h2>💡 الاستنتاجات والتوصيات الرئيسية</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <h3 style='color: #2c3e50;'>📊 الاستنتاجات الرئيسية:</h3>
        <ul style='color: #34495e; line-height: 1.8;'>
            <li><strong>انخفاض تدريجي:</strong> انخفاض ملحوظ في عدد الحوادث من ١٤٣٧ إلى ١٤٣٩ هـ</li>
            <li><strong>شهر رمضان:</strong> يسجل أعلى معدلات الحوادث سنوياً</li>
            <li><strong>الذكور:</strong> يشكلون ~٨٣٪ من المصابين و~٨٧٪ من الوفيات</li>
            <li><strong>المدن الكبرى:</strong> الرياض وجدة ومكة تسجل أعلى الأرقام</li>
            <li><strong>خارج المدن:</strong> حوادث أكثر خطورة بمعدل وفيات أعلى</li>
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
            <li><strong>الشباب:</strong> برامج توعية مستهدفة للفئة العمرية ١٨-٣٠</li>
            <li><strong>التكنولوجيا:</strong> استخدام الذكاء الاصطناعي للتنبؤ بالنقاط الخطرة</li>
            <li><strong>المراقبة:</strong> زيادة الكاميرات والرادارات في المناطق عالية الخطورة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================
# القسم 7: جدول البيانات التفصيلي
# ============================
st.markdown("""
<div class="section-header">
    <h2>📋 جدول البيانات التفصيلي</h2>
</div>
""", unsafe_allow_html=True)

# عرض عينة من البيانات
display_df = filtered_df.groupby(['year', 'city']).agg({
    'total_cases': 'sum',
    'total_injured': 'sum',
    'total_dead': 'sum',
    'fatality_rate': 'mean'
}).reset_index()

display_df = display_df.sort_values('total_cases', ascending=False)
display_df.columns = ['السنة', 'المدينة', 'إجمالي الحوادث', 'إجمالي المصابين', 'إجمالي الوفيات', 'معدل الوفيات %']

st.dataframe(
    display_df.style.background_gradient(cmap='Reds', subset=['إجمالي الحوادث', 'إجمالي الوفيات'])
                   .format({'معدل الوفيات %': '{:.2f}%'}),
    use_container_width=True,
    height=400
)

# ============================
# التذييل
# ============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p style='font-size: 16px;'>© ٢٠٢٦ الإدارة العامة للمرور - وزارة الداخلية | المملكة العربية السعودية</p>
    <p style='font-size: 14px; opacity: 0.8;'>تم التطوير باستخدام Streamlit و Plotly</p>
</div>
""", unsafe_allow_html=True)
