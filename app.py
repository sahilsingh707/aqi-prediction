
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Load models ──────────────────────────────────────────────────────────────
with open('aqi_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)
with open('aqi_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('city_day.csv')
cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata', 'Chennai', 'Lucknow']
df = df_raw[df_raw['City'].isin(cities)].copy()
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']
city_defaults = df.groupby('City')[features].mean().round(1)

# ── WHO safe limits ───────────────────────────────────────────────────────────
WHO_LIMITS = {
    'PM2.5': 15, 'PM10': 45, 'NO': 40, 'NO2': 25,
    'NOx': 50,   'NH3': 100, 'CO': 4,  'SO2': 40,
    'O3': 100,   'Benzene': 1.7, 'Toluene': 260
}

# ── Health risk data ──────────────────────────────────────────────────────────
HEALTH_DATA = {
    'Good': {
        'color': '#00e676', 'emoji': '🟢',
        'risk': 'Air quality is satisfactory. Little to no risk.',
        'affected': 'No specific groups at risk.',
        'symptoms': 'None expected.',
        'advice': [
            '✅ Safe to go outside and exercise',
            '✅ Windows can be kept open',
            '✅ No mask required',
            '✅ All outdoor activities are fine'
        ]
    },
    'Satisfactory': {
        'color': '#aeea00', 'emoji': '🟡',
        'risk': 'Acceptable air quality. Minor risk for very sensitive individuals.',
        'affected': 'People with severe respiratory conditions.',
        'symptoms': 'Mild breathing discomfort for highly sensitive people.',
        'advice': [
            '✅ Safe for most people to go outside',
            '⚠️ Sensitive individuals should limit prolonged outdoor exertion',
            '✅ Light outdoor activities are fine',
            '✅ No mask needed for healthy individuals'
        ]
    },
    'Moderate': {
        'color': '#ffd600', 'emoji': '🟠',
        'risk': 'Members of sensitive groups may experience health effects.',
        'affected': 'Children, elderly, people with asthma or heart disease.',
        'symptoms': 'Coughing, throat irritation, shortness of breath.',
        'advice': [
            '⚠️ Sensitive groups should reduce outdoor activity',
            '⚠️ Avoid prolonged outdoor exercise',
            '😷 Sensitive individuals should consider wearing a mask',
            '🏠 Keep windows closed during peak hours'
        ]
    },
    'Poor': {
        'color': '#ff6d00', 'emoji': '🔴',
        'risk': 'Everyone may begin to experience health effects.',
        'affected': 'Everyone, especially children, elderly, and those with lung/heart disease.',
        'symptoms': 'Chest tightness, coughing, wheezing, eye irritation, headaches.',
        'advice': [
            '🚫 Avoid all outdoor physical activity',
            '😷 Wear N95 mask if going outside',
            '🏠 Stay indoors with windows shut',
            '💊 Asthma/heart patients must carry medication',
            '🌬️ Use air purifier indoors if available'
        ]
    },
    'Very Poor': {
        'color': '#dd2c00', 'emoji': '🟣',
        'risk': 'Serious health effects for everyone. Emergency conditions for sensitive groups.',
        'affected': 'Everyone. Critical risk for elderly, children, and patients.',
        'symptoms': 'Severe breathing difficulty, chest pain, fatigue, dizziness, skin irritation.',
        'advice': [
            '🚫 Do NOT go outside unless absolutely necessary',
            '😷 N95/N99 mask mandatory if outside',
            '🏠 Seal windows and doors',
            '🌬️ Run air purifier on highest setting',
            '🏥 Seek medical help if experiencing symptoms',
            '📵 Cancel all outdoor events'
        ]
    },
    'Severe': {
        'color': '#8b0000', 'emoji': '⚫',
        'risk': '🚨 HAZARDOUS — Health emergency. Entire population is likely to be affected.',
        'affected': 'Everyone. Life-threatening for elderly, children, and patients.',
        'symptoms': 'Severe chest pain, breathing failure, heart palpitations, unconsciousness risk.',
        'advice': [
            '🚨 STAY INDOORS — This is a health emergency',
            '🚫 Absolutely no outdoor activity',
            '😷 N99 respirator required if emergency exit needed',
            '🏥 Hospitals on alert — seek help immediately for any symptoms',
            '🌬️ Maximum air purification indoors',
            '📞 Follow government health advisories'
        ]
    }
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI Predictor India", page_icon="🌿", layout="wide")

st.markdown("""
    <h1 style='text-align:center; color:#00e676; font-size:2.5rem;'>
        🌿 Air Quality Index Predictor
    </h1>
    <p style='text-align:center; color:#a0a0a0; font-size:16px;'>
        Predict AQI · Assess Health Risks · Compare Cities · Analyse Trends
    </p>
    <hr style='border:1px solid #3d4166;'>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict AQI", "📊 City Comparison", "📈 Trend Analysis"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    col_city, col_blank = st.columns([1, 2])
    with col_city:
        selected_city = st.selectbox(
            "🏙️ Select city to auto-fill average values:",
            options=["Custom"] + cities
        )

    if selected_city != "Custom":
        defaults = city_defaults.loc[selected_city]
    else:
        defaults = pd.Series({f: 0.0 for f in features})

    st.markdown("#### Enter Pollutant Concentrations")
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25    = st.number_input("PM2.5 (µg/m³)",   min_value=0.0, value=float(defaults['PM2.5']), step=1.0)
        pm10    = st.number_input("PM10 (µg/m³)",    min_value=0.0, value=float(defaults['PM10']),  step=1.0)
        no      = st.number_input("NO (µg/m³)",      min_value=0.0, value=float(defaults['NO']),    step=1.0)
        no2     = st.number_input("NO2 (µg/m³)",     min_value=0.0, value=float(defaults['NO2']),   step=1.0)
    with col2:
        nox     = st.number_input("NOx (µg/m³)",     min_value=0.0, value=float(defaults['NOx']),   step=1.0)
        nh3     = st.number_input("NH3 (µg/m³)",     min_value=0.0, value=float(defaults['NH3']),   step=1.0)
        co      = st.number_input("CO (mg/m³)",      min_value=0.0, value=float(defaults['CO']),    step=0.1)
        so2     = st.number_input("SO2 (µg/m³)",     min_value=0.0, value=float(defaults['SO2']),   step=1.0)
    with col3:
        o3      = st.number_input("O3 (µg/m³)",      min_value=0.0, value=float(defaults['O3']),    step=1.0)
        benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, value=float(defaults['Benzene']),step=0.1)
        toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, value=float(defaults['Toluene']),step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict AQI & Health Risk", use_container_width=True):

        input_data = pd.DataFrame([{
            'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox,
            'NH3': nh3, 'CO': co, 'SO2': so2, 'O3': o3,
            'Benzene': benzene, 'Toluene': toluene
        }])

        aqi_pred      = regressor.predict(input_data)[0]
        category_pred = classifier.predict(input_data)[0]
        health        = HEALTH_DATA[category_pred]
        color         = health['color']
        city_label    = f" — {selected_city}" if selected_city != "Custom" else ""

        # ── AQI Result ────────────────────────────────────────────────────────
        st.markdown("<hr style='border:1px solid #3d4166;'>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
                <div style='background:#1a1d27; border-radius:12px; padding:30px;
                            text-align:center; border:2px solid {color};'>
                    <h3 style='color:#a0a0a0; margin:0;'>Predicted AQI{city_label}</h3>
                    <h1 style='color:{color}; font-size:72px; margin:10px 0;'>{aqi_pred:.1f}</h1>
                </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
                <div style='background:#1a1d27; border-radius:12px; padding:30px;
                            text-align:center; border:2px solid {color};'>
                    <h3 style='color:#a0a0a0; margin:0;'>Category</h3>
                    <h1 style='font-size:52px; margin:8px 0;'>{health['emoji']}</h1>
                    <h2 style='color:{color}; margin:0;'>{category_pred}</h2>
                </div>
            """, unsafe_allow_html=True)

        # ── AQI Gauge ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
        ax.set_facecolor('#0f1117')

        segments = [
            (0,   50,  '#00e676', 'Good'),
            (50,  100, '#aeea00', 'Satisfactory'),
            (100, 200, '#ffd600', 'Moderate'),
            (200, 300, '#ff6d00', 'Poor'),
            (300, 400, '#dd2c00', 'Very Poor'),
            (400, 500, '#8b0000', 'Severe'),
        ]

        for start, end, clr, label in segments:
            ax.barh(0, end - start, left=start, height=0.4,
                    color=clr, alpha=0.85)
            ax.text((start + end) / 2, 0.35, label,
                    ha='center', va='bottom', color='white', fontsize=8)

        marker_x = min(aqi_pred, 499)
        ax.annotate('', xy=(marker_x, -0.15), xytext=(marker_x, 0.18),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
        ax.text(marker_x, -0.28, f'{aqi_pred:.1f}',
                ha='center', color='white', fontsize=12, fontweight='bold')

        ax.set_xlim(0, 500)
        ax.set_ylim(-0.5, 0.7)
        ax.axis('off')
        ax.set_title('AQI Scale Meter', color='white', fontsize=13, pad=10)
        st.pyplot(fig_gauge)

        # ── Health Risk Panel ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='background:#1a1d27; border-radius:12px; padding:24px;
                        border-left:5px solid {color};'>
                <h2 style='color:{color}; margin-top:0;'>
                    {health['emoji']} Health Risk Assessment
                </h2>
                <table style='width:100%; color:#e0e0e0; font-size:15px;
                              border-collapse:collapse;'>
                    <tr style='border-bottom:1px solid #3d4166;'>
                        <td style='padding:10px; color:#a0a0a0; width:30%;'>
                            ⚠️ Risk Level
                        </td>
                        <td style='padding:10px;'>{health['risk']}</td>
                    </tr>
                    <tr style='border-bottom:1px solid #3d4166;'>
                        <td style='padding:10px; color:#a0a0a0;'>
                            👥 Most Affected
                        </td>
                        <td style='padding:10px;'>{health['affected']}</td>
                    </tr>
                    <tr>
                        <td style='padding:10px; color:#a0a0a0;'>
                            🤒 Expected Symptoms
                        </td>
                        <td style='padding:10px;'>{health['symptoms']}</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:{color};'>📋 Recommendations</h3>",
                    unsafe_allow_html=True)

        rec_cols = st.columns(2)
        for i, rec in enumerate(health['advice']):
            with rec_cols[i % 2]:
                st.markdown(f"""
                    <div style='background:#1a1d27; border-radius:8px; padding:12px;
                                margin-bottom:8px; border:1px solid #3d4166;
                                color:#e0e0e0; font-size:14px;'>
                        {rec}
                    </div>
                """, unsafe_allow_html=True)

        # ── WHO Pollutant Safety Check ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:white;'>🔬 Pollutant Safety Check (vs WHO Limits)</h3>",
                    unsafe_allow_html=True)

        user_values = {
            'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox,
            'NH3': nh3, 'CO': co, 'SO2': so2, 'O3': o3,
            'Benzene': benzene, 'Toluene': toluene
        }

        who_cols = st.columns(4)
        for i, (poll, val) in enumerate(user_values.items()):
            limit  = WHO_LIMITS[poll]
            safe   = val <= limit
            clr2   = '#00e676' if safe else '#ef5350'
            status = '✅ Safe' if safe else '❌ Exceeds Limit'
            pct    = (val / limit) * 100

            with who_cols[i % 4]:
                st.markdown(f"""
                    <div style='background:#1a1d27; border-radius:8px; padding:12px;
                                margin-bottom:10px; border-top:3px solid {clr2};
                                text-align:center;'>
                        <b style='color:{clr2};'>{poll}</b><br>
                        <span style='color:white; font-size:20px;'>{val}</span><br>
                        <span style='color:#a0a0a0; font-size:11px;'>
                            WHO limit: {limit}
                        </span><br>
                        <span style='color:{clr2}; font-size:12px;'>{status}</span><br>
                        <span style='color:#a0a0a0; font-size:11px;'>
                            {pct:.0f}% of limit
                        </span>
                    </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — CITY COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Compare AQI Across Indian Cities")

    city_stats = df.groupby('City')['AQI'].agg(['mean','median','max','min']).round(1)
    city_stats.columns = ['Average AQI', 'Median AQI', 'Max AQI', 'Min AQI']
    city_stats = city_stats.reindex(cities)

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(12, 5), facecolor='#0f1117')
    ax2.set_facecolor('#1a1d27')
    clrs = ['#ef5350','#42a5f5','#66bb6a','#ffa726','#ab47bc','#26c6da']
    bars = ax2.bar(city_stats.index, city_stats['Average AQI'],
                   color=clrs, width=0.5)
    for bar, val in zip(bars, city_stats['Average AQI']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.0f}', ha='center', color='white', fontsize=11,
                 fontweight='bold')
    ax2.set_title('Average AQI by City', color='white', fontsize=14,
                  fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines[['top','right','left','bottom']].set_visible(False)
    ax2.set_facecolor('#1a1d27')
    st.pyplot(fig2)

    # Stats table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Detailed City Statistics")
    st.dataframe(
        city_stats.style.background_gradient(cmap='RdYlGn_r', subset=['Average AQI']),
        use_container_width=True
    )

    # Season heatmap
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🌡️ Season-wise AQI Heatmap")

    season_map = {12:'Winter',1:'Winter',2:'Winter',
                  3:'Summer',4:'Summer',5:'Summer',
                  6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
                  10:'Post-Monsoon',11:'Post-Monsoon'}
    df['Season'] = df['Date'].dt.month.map(season_map)
    pivot = df.groupby(['City','Season'])['AQI'].mean().unstack()
    pivot = pivot[['Winter','Post-Monsoon','Summer','Monsoon']].reindex(cities)

    fig3, ax3 = plt.subplots(figsize=(10, 5), facecolor='#0f1117')
    ax3.set_facecolor('#0f1117')
    im = ax3.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(pivot.columns, color='white', fontsize=11)
    ax3.set_yticks(range(len(cities)))
    ax3.set_yticklabels(pivot.index, color='white', fontsize=11)
    for i in range(len(cities)):
        for j in range(4):
            ax3.text(j, i, f'{pivot.values[i,j]:.0f}',
                     ha='center', va='center', color='white',
                     fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax3, label='Avg AQI')
    ax3.set_title('Average AQI by City & Season', color='white',
                  fontsize=13, fontweight='bold')
    st.pyplot(fig3)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — TREND ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Year-wise AQI Trend Analysis")

    trend_city = st.selectbox("Select city for trend:", cities, key='trend')

    city_trend = df[df['City'] == trend_city].copy()
    city_trend = city_trend.sort_values('Date')

    fig4, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='#0f1117')

    # Full trend
    ax_t = axes[0]
    ax_t.set_facecolor('#1a1d27')
    ax_t.plot(city_trend['Date'], city_trend['AQI'],
              color='#42a5f5', linewidth=0.8, alpha=0.7)
    ax_t.fill_between(city_trend['Date'], city_trend['AQI'],
                      alpha=0.15, color='#42a5f5')

    # AQI danger lines
    for level, clr, lbl in [(200,'#ffd600','Moderate'),
                             (300,'#ff6d00','Poor'),
                             (400,'#ef5350','Very Poor')]:
        ax_t.axhline(level, color=clr, linewidth=1,
                     linestyle='--', alpha=0.6, label=lbl)

    ax_t.set_title(f'{trend_city} — Daily AQI Trend',
                   color='white', fontsize=13, fontweight='bold')
    ax_t.tick_params(colors='white')
    ax_t.spines[['top','right']].set_visible(False)
    ax_t.legend(fontsize=9, framealpha=0.2)
    ax_t.set_ylabel('AQI', color='white')

    # Yearly average
    ax_y = axes[1]
    ax_y.set_facecolor('#1a1d27')
    yearly = city_trend.groupby('Year')['AQI'].mean()
    ax_y.bar(yearly.index, yearly.values, color='#42a5f5', alpha=0.8, width=0.6)
    for x, v in zip(yearly.index, yearly.values):
        ax_y.text(x, v + 2, f'{v:.0f}', ha='center',
                  color='white', fontsize=10, fontweight='bold')
    ax_y.set_title(f'{trend_city} — Yearly Average AQI',
                   color='white', fontsize=13, fontweight='bold')
    ax_y.tick_params(colors='white')
    ax_y.spines[['top','right']].set_visible(False)
    ax_y.set_ylabel('Avg AQI', color='white')

    plt.tight_layout(pad=2)
    st.pyplot(fig4)

    # Worst days
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### 🚨 Top 10 Worst AQI Days in {trend_city}")
    worst = city_trend.nlargest(10, 'AQI')[['Date','AQI']].reset_index(drop=True)
    worst['Date'] = worst['Date'].dt.strftime('%d %b %Y')
    worst.index += 1
    st.dataframe(worst, use_container_width=True)

print("app.py written ✅")
