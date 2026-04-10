"""
Sentinel — Public Health Intelligence Dashboard
Streamlit in Snowflake (SiS) Application
Packages needed: plotly
Python: 3.10
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
from snowflake.snowpark.context import get_active_session

# ── Session ──────────────────────────────────────────────────
session = get_active_session()

def query(sql):
    """Run SQL and return pandas DataFrame."""
    return session.sql(sql).to_pandas()

def to_date(s):
    """Convert string to datetime.date for safe comparison."""
    return pd.Timestamp(s).date()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="Sentinel", page_icon="◆", layout="wide")

# ── Design System ────────────────────────────────────────────
MAROON = "#800020"
MAROON_LIGHT = "#9A1B3A"
WHITE = "#FFFFFF"
GRAY_100 = "#F3F0ED"
GRAY_300 = "#D1CBC4"
GRAY_500 = "#8C8278"
GRAY_700 = "#4A4540"
GRAY_900 = "#1E1B18"
CHART_BLUE = "#2563EB"
CHART_RED = "#DC2626"
CHART_GREEN = "#059669"
CHART_AMBER = "#D97706"
TIER_COLORS = {'HIGH': CHART_RED, 'MODERATE': CHART_AMBER, 'LOW': CHART_GREEN}

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"] {{ font-family: 'Source Sans 3', sans-serif; color: {GRAY_900}; }}
.main .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}
section[data-testid="stSidebar"] {{ background: {MAROON}; }}
section[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
section[data-testid="stSidebar"] .stSelectbox label {{
    color: {GRAY_300} !important; font-weight: 600; font-size: .85rem;
    text-transform: uppercase; letter-spacing: .5px;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.15); }}
.dash-title {{ font-family: 'Playfair Display', serif; font-size: 2.2rem;
    font-weight: 700; color: {MAROON}; margin: 0; line-height: 1.2; }}
.dash-subtitle {{ font-size: 1rem; color: {GRAY_500}; margin: 4px 0 1.2rem 0;
    border-bottom: 2px solid {MAROON}; padding-bottom: 12px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 2px solid {GRAY_100}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 20px; font-weight: 600; font-size: .9rem;
    color: {GRAY_500}; border-bottom: 3px solid transparent; }}
.stTabs [aria-selected="true"] {{ color: {MAROON} !important; border-bottom-color: {MAROON} !important; }}
.sec-hdr {{ font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700;
    color: {MAROON}; margin: 1.5rem 0 .75rem 0; padding-bottom: 6px;
    border-bottom: 1px solid {GRAY_300}; }}
.sub-hdr {{ font-size: 1rem; font-weight: 600; color: {GRAY_700}; margin: 1rem 0 .5rem 0; }}
.kpi-card {{ background: {WHITE}; border: 1px solid {GRAY_100}; border-radius: 8px;
    padding: 16px 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
.kpi-val {{ font-size: 1.8rem; font-weight: 700; color: {MAROON}; line-height: 1.1; }}
.kpi-label {{ font-size: .75rem; color: {GRAY_500}; text-transform: uppercase;
    letter-spacing: .4px; margin-top: 4px; }}
.tier-high {{ display: inline-block; background: #FEE2E2; color: #991B1B;
    padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: .85rem; }}
.tier-mod {{ display: inline-block; background: #FEF3C7; color: #92400E;
    padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: .85rem; }}
.tier-low {{ display: inline-block; background: #D1FAE5; color: #065F46;
    padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: .85rem; }}
.footer {{ text-align: center; color: {GRAY_500}; font-size: .8rem;
    padding: 1.5rem 0 .5rem 0; border-top: 1px solid {GRAY_100}; margin-top: 2rem; }}
</style>""", unsafe_allow_html=True)


# ── Load Data ────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_all():
    risk = query("SELECT * FROM PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS ORDER BY RISK_SCORE DESC")
    features = query("SELECT * FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES ORDER BY COUNTRY_REGION, DATE")
    hf = query("SELECT * FROM PUBLIC_HEALTH_DB.ML.HISTORY_FORECAST ORDER BY COUNTRY_REGION, DATE")
    forecast = query("SELECT * FROM PUBLIC_HEALTH_DB.ML.FORECASTS ORDER BY COUNTRY_REGION, DATE")
    anomalies = query("SELECT * FROM PUBLIC_HEALTH_DB.ML.ANOMALIES ORDER BY COUNTRY_REGION, DATE")

    # Normalize dates — Snowflake returns datetime.date, convert to pd.Timestamp
    for df in [features, hf, forecast, anomalies]:
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])

    # Narratives — may not exist yet
    try:
        narratives = query("SELECT * FROM PUBLIC_HEALTH_DB.ANALYTICS.CORTEX_NARRATIVES ORDER BY COUNTRY_REGION")
    except Exception:
        narratives = pd.DataFrame()
    try:
        anomaly_expl = query("SELECT * FROM PUBLIC_HEALTH_DB.ANALYTICS.ANOMALY_EXPLANATIONS ORDER BY COUNTRY_REGION, DATE")
        if 'DATE' in anomaly_expl.columns:
            anomaly_expl['DATE'] = pd.to_datetime(anomaly_expl['DATE'])
    except Exception:
        anomaly_expl = pd.DataFrame()
    try:
        global_triage = query("SELECT GLOBAL_BRIEF FROM PUBLIC_HEALTH_DB.ANALYTICS.GLOBAL_TRIAGE")
    except Exception:
        global_triage = pd.DataFrame()
    try:
        vaccinations = query("SELECT * FROM PUBLIC_HEALTH_DB.RAW.VACCINATIONS ORDER BY COUNTRY_REGION, DATE")
        if 'DATE' in vaccinations.columns and not vaccinations.empty:
            vaccinations['DATE'] = pd.to_datetime(vaccinations['DATE'])
    except Exception:
        vaccinations = pd.DataFrame()
    try:
        metrics_df = query("SELECT * FROM PUBLIC_HEALTH_DB.ML.FORECAST_METRICS")
    except Exception:
        metrics_df = pd.DataFrame()

    return {
        'risk': risk, 'features': features, 'hf': hf, 'forecast': forecast,
        'anomalies': anomalies, 'narratives': narratives,
        'anomaly_expl': anomaly_expl, 'global_triage': global_triage,
        'vaccinations': vaccinations, 'metrics': metrics_df
    }

data = load_all()
risk = data['risk']
features = data['features']
hf = data['hf']
forecast = data['forecast']
anomalies = data['anomalies']
narratives = data['narratives']
anomaly_expl = data['anomaly_expl']
global_triage = data['global_triage']
vaccinations = data['vaccinations']
metrics_df = data['metrics']

countries = risk['COUNTRY_REGION'].tolist()

# Guard — if risk table is empty somehow
if len(countries) == 0:
    st.error("No countries found in RISK_TIERS table. Run 04_risk_and_cortex.sql first.")
    st.stop()


# ── Helpers ──────────────────────────────────────────────────
def tier_badge(t):
    css = {'HIGH': 'tier-high', 'MODERATE': 'tier-mod', 'LOW': 'tier-low'}.get(str(t), '')
    return f'<span class="{css}">{t}</span>'

def plotly_layout(fig, height=380):
    fig.update_layout(
        template='plotly_white', height=height,
        font=dict(family='Source Sans 3', size=13),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.15),
        hovermode='x unified'
    )
    return fig

def safe_val(series, col, fmt="{:.2f}", default="—"):
    v = series.get(col) if isinstance(series, dict) else getattr(series, col, None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return fmt.format(v)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 8px 0;">
        <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700;">
            Sentinel
        </div>
        <div style="font-size: .8rem; opacity: .7; margin-top: 2px;">
            Epidemiological Intelligence
        </div>
    </div>
    """, True)
    st.markdown("---")

    sel_country = st.selectbox("COUNTRY", countries, index=0)

    cr = risk[risk.COUNTRY_REGION == sel_country].iloc[0]
    rt_val = safe_val(cr, 'RT_EFFECTIVE', "{:.2f}")
    score_val = int(cr.RISK_SCORE) if pd.notna(cr.RISK_SCORE) else 0

    st.markdown(f"""
    <div style="background: rgba(255,255,255,.1); border-radius: 6px; padding: 12px; margin-top: 12px;">
        <div style="font-size: .75rem; text-transform: uppercase; opacity: .6;">Risk Assessment</div>
        <div style="font-size: 1.6rem; font-weight: 700; margin: 4px 0;">{cr.RISK_TIER}</div>
        <div style="font-size: .85rem;">Score: {score_val}/100</div>
    </div>
    """, True)

    # Data gap warning
    data_gap = bool(cr.DATA_GAP) if 'DATA_GAP' in cr.index and pd.notna(cr.DATA_GAP) else False
    if data_gap:
        st.markdown("""
        <div style="background: rgba(251,191,36,.15); border: 1px solid rgba(251,191,36,.4);
                    border-radius: 6px; padding: 8px 10px; margin-top: 8px; font-size: .8rem;">
            ⚠️ <b>Data Gap</b><br>This country stopped submitting COVID reports. Showing last available data.
        </div>""", True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: .8rem; opacity: .6; line-height: 1.6;">
        Data: JHU CSSE COVID-19<br>
        Models: Snowflake ML<br>
        AI: Snowflake Cortex<br>
        Countries: {len(countries)}
    </div>
    """, True)


# ── Header ───────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('<div class="dash-title">Sentinel</div>', True)
    st.markdown(f'<div class="dash-subtitle">Epidemiological Intelligence Dashboard — '
                f'{len(countries)} Countries</div>', True)
with c2:
    st.markdown(f'<div style="text-align:right; padding-top:12px;">'
                f'<span style="font-size:.85rem; color:{GRAY_500};">Selected: </span>'
                f'<span style="font-size:1.1rem; font-weight:700; color:{MAROON};">{sel_country}</span>'
                f'<br>{tier_badge(cr.RISK_TIER)}</div>', True)


# ── Tabs ─────────────────────────────────────────────────────
tabs = st.tabs(["Overview", "Epidemiology", "Forecast", "Intelligence",
                "Comparison", "Anomalies", "Methodology"])


# ═══════════ TAB 1: OVERVIEW ═════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-hdr">Global Risk Overview</div>', True)

    high_n = len(risk[risk.RISK_TIER == 'HIGH'])
    mod_n = len(risk[risk.RISK_TIER == 'MODERATE'])
    low_n = len(risk[risk.RISK_TIER == 'LOW'])
    avg_rt = risk.RT_EFFECTIVE.dropna().mean()
    avg_score = risk.RISK_SCORE.mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, val, label in [
        (k1, str(len(countries)), "Countries"),
        (k2, f"{avg_score:.0f}" if pd.notna(avg_score) else "—", "Avg Risk Score"),
        (k3, f"{avg_rt:.2f}" if pd.notna(avg_rt) else "—", "Global Avg Rt"),
        (k4, str(high_n), "High Risk"),
        (k5, str(low_n), "Low Risk"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                        f'<div class="kpi-label">{label}</div></div>', True)

    st.markdown("")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = px.bar(risk.sort_values('RISK_SCORE'), x='RISK_SCORE', y='COUNTRY_REGION',
                     color='RISK_TIER', orientation='h', color_discrete_map=TIER_COLORS)
        fig.update_layout(xaxis_title='Risk Score', yaxis_title='', showlegend=True, legend_title_text='')
        plotly_layout(fig, 460)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        scatter_df = risk.dropna(subset=['RT_EFFECTIVE', 'RISK_SCORE', 'ROLLING_AVG_7D_CASES'])
        if not scatter_df.empty:
            fig2 = px.scatter(scatter_df, x='RT_EFFECTIVE', y='RISK_SCORE', color='RISK_TIER',
                              size='ROLLING_AVG_7D_CASES', hover_name='COUNTRY_REGION',
                              color_discrete_map=TIER_COLORS)
            fig2.add_vline(x=1.0, line_dash='dash', line_color=CHART_RED,
                           annotation_text='Rt = 1.0', annotation_font_size=11)
            fig2.update_layout(xaxis_title='Rt', yaxis_title='Risk Score', legend_title_text='')
            plotly_layout(fig2, 460)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sub-hdr">All Countries</div>', True)
    display_cols = ['COUNTRY_REGION', 'RISK_TIER', 'RISK_SCORE', 'ROLLING_AVG_7D_CASES',
                    'RT_EFFECTIVE', 'DOUBLING_TIME_DAYS', 'EPIDEMIC_PHASE', 'FORECAST_TREND_PCT']
    avail_cols = [c for c in display_cols if c in risk.columns]
    tbl = risk[avail_cols].copy()
    rename_map = {'COUNTRY_REGION': 'Country', 'RISK_TIER': 'Risk', 'RISK_SCORE': 'Score',
                  'ROLLING_AVG_7D_CASES': '7-Day Avg', 'RT_EFFECTIVE': 'Rt',
                  'DOUBLING_TIME_DAYS': 'Doubling (d)', 'EPIDEMIC_PHASE': 'Phase',
                  'FORECAST_TREND_PCT': 'Forecast %'}
    tbl = tbl.rename(columns={k: v for k, v in rename_map.items() if k in tbl.columns})
    tbl = tbl.fillna("—")
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════ TAB 2: EPIDEMIOLOGY ═════════════════════════════
with tabs[1]:
    st.markdown(f'<div class="sec-hdr">Epidemiological Profile — {sel_country}</div>', True)

    DATE_START = pd.Timestamp('2020-06-01')
    feat = features[(features.COUNTRY_REGION == sel_country) &
                    (features.DATE >= DATE_START)].sort_values('DATE')

    data_gap = bool(cr.DATA_GAP) if 'DATA_GAP' in cr.index and pd.notna(cr.DATA_GAP) else False
    if data_gap:
        st.warning(f"⚠️ **Data Gap**: {sel_country} stopped reporting COVID-19 data before the dataset end date. "
                   f"Metrics below reflect the last available reporting period.")

    if feat.empty:
        st.warning("No feature data available for this country.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rt", safe_val(cr, 'RT_EFFECTIVE'))
        m2.metric("7-Day Avg Cases", safe_val(cr, 'ROLLING_AVG_7D_CASES', "{:,.0f}"))
        phase_map = {'EXPONENTIAL_GROWTH': 'Exp. Growth', 'EXPONENTIAL_DECLINE': 'Exp. Decline',
                     'LINEAR_GROWTH': 'Lin. Growth', 'LINEAR_DECLINE': 'Lin. Decline', 'PLATEAU': 'Plateau'}
        phase_str = phase_map.get(str(cr.EPIDEMIC_PHASE), str(cr.EPIDEMIC_PHASE).replace('_', ' ').title()) if pd.notna(cr.EPIDEMIC_PHASE) else "—"
        m3.metric("Phase", phase_str)
        m4.metric("CFR", safe_val(cr, 'CASE_FATALITY_RATE', "{:.2f}%"))

        # Rt timeline
        rt_data = feat.dropna(subset=['RT_EFFECTIVE'])
        if not rt_data.empty:
            st.markdown('<div class="sub-hdr">Effective Reproduction Number (Rt)</div>', True)
            fig_rt = go.Figure()
            fig_rt.add_trace(go.Scatter(x=rt_data.DATE, y=rt_data.RT_EFFECTIVE, mode='lines',
                                        name='Rt', line=dict(color=CHART_BLUE, width=2)))
            fig_rt.add_hline(y=1.0, line_dash='dash', line_color=CHART_RED,
                             annotation_text='Epidemic Threshold')
            plotly_layout(fig_rt, 340)
            fig_rt.update_layout(yaxis_title='Rt')
            st.plotly_chart(fig_rt, use_container_width=True)

        # Cases + Deaths
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sub-hdr">Daily Cases (Rolling Average)</div>', True)
            fig_c = go.Figure()
            if 'ROLLING_AVG_7D_CASES' in feat.columns:
                fig_c.add_trace(go.Scatter(x=feat.DATE, y=feat.ROLLING_AVG_7D_CASES,
                                           fill='tozeroy', name='7-Day',
                                           line=dict(color=CHART_BLUE),
                                           fillcolor='rgba(37,99,235,.08)'))
            if 'ROLLING_AVG_14D_CASES' in feat.columns:
                fig_c.add_trace(go.Scatter(x=feat.DATE, y=feat.ROLLING_AVG_14D_CASES,
                                           name='14-Day', line=dict(color='#6B7280', dash='dash')))
            plotly_layout(fig_c, 300)
            st.plotly_chart(fig_c, use_container_width=True)
        with c2:
            st.markdown('<div class="sub-hdr">Daily Deaths (Rolling Average)</div>', True)
            fig_d = go.Figure()
            if 'ROLLING_AVG_7D_DEATHS' in feat.columns:
                fig_d.add_trace(go.Scatter(x=feat.DATE, y=feat.ROLLING_AVG_7D_DEATHS,
                                           fill='tozeroy', name='7-Day',
                                           line=dict(color=CHART_RED),
                                           fillcolor='rgba(220,38,38,.08)'))
            plotly_layout(fig_d, 300)
            st.plotly_chart(fig_d, use_container_width=True)

        # Vaccination
        if not vaccinations.empty:
            vax_c = vaccinations[vaccinations.COUNTRY_REGION == sel_country].sort_values('DATE')
            if not vax_c.empty and 'VACCINATED_PER_100' in vax_c.columns and vax_c.VACCINATED_PER_100.notna().any():
                st.markdown('<div class="sub-hdr">Vaccination Coverage</div>', True)
                fig_vax = go.Figure()
                fig_vax.add_trace(go.Scatter(x=vax_c.DATE, y=vax_c.VACCINATED_PER_100,
                                             name='1+ Dose', fill='tozeroy',
                                             line=dict(color=CHART_BLUE),
                                             fillcolor='rgba(37,99,235,.06)'))
                if 'FULLY_VACCINATED_PER_100' in vax_c.columns:
                    fig_vax.add_trace(go.Scatter(x=vax_c.DATE, y=vax_c.FULLY_VACCINATED_PER_100,
                                                 name='Fully Vaccinated',
                                                 line=dict(color=CHART_GREEN, dash='dash')))
                plotly_layout(fig_vax, 300)
                fig_vax.update_layout(yaxis_title='% of Population')
                st.plotly_chart(fig_vax, use_container_width=True)


# ═══════════ TAB 3: FORECAST ═════════════════════════════════
with tabs[2]:
    st.markdown(f'<div class="sec-hdr">30-Day Forecast — {sel_country}</div>', True)

    country_hf = hf[hf.COUNTRY_REGION == sel_country].sort_values('DATE')
    all_actual = country_hf[country_hf.TYPE == 'ACTUAL'].copy()
    fcast = country_hf[country_hf.TYPE == 'FORECAST'].copy()

    # Clamp negative values to zero for display
    if not all_actual.empty and 'VAL' in all_actual.columns:
        all_actual['VAL'] = all_actual['VAL'].clip(lower=0)
    if not fcast.empty and 'VAL' in fcast.columns:
        fcast['VAL'] = fcast['VAL'].clip(lower=0)

    # Split actual data into history (before forecast) and validation (during forecast)
    if not fcast.empty:
        forecast_start = fcast.DATE.min()
        forecast_end = fcast.DATE.max()
        # Historical data leading up to forecast (120 days for context)
        history = all_actual[all_actual.DATE < forecast_start].tail(120)
        # Ground truth during forecast window (to compare accuracy)
        validation = all_actual[(all_actual.DATE >= forecast_start) & (all_actual.DATE <= forecast_end)]
    else:
        history = all_actual.tail(180)
        validation = pd.DataFrame()

    # Forecast KPIs — displayed as columns above the chart
    fc_c = forecast[forecast.COUNTRY_REGION == sel_country] if not forecast.empty else pd.DataFrame()
    km1, km2, km3, km4 = st.columns(4)
    km1.metric("Current 7-Day Avg", safe_val(cr, 'ROLLING_AVG_7D_CASES', "{:,.0f}"))
    if not fc_c.empty and 'FORECASTED_CASES' in fc_c.columns:
        fc_end_v = max(0, fc_c.FORECASTED_CASES.iloc[-1]) if pd.notna(fc_c.FORECASTED_CASES.iloc[-1]) else 0
        fc_peak_v = max(0, fc_c.FORECASTED_CASES.max()) if pd.notna(fc_c.FORECASTED_CASES.max()) else 0
        km2.metric("Forecast End", f"{fc_end_v:,.0f}")
        km3.metric("Forecast Peak", f"{fc_peak_v:,.0f}")
    elif not fcast.empty:
        km2.metric("Forecast End", f"{fcast.VAL.iloc[-1]:,.0f}")
        km3.metric("Forecast Peak", f"{fcast.VAL.max():,.0f}")
    else:
        km2.metric("Forecast End", "—")
        km3.metric("Forecast Peak", "—")
    try:
        ftp = cr.FORECAST_TREND_PCT
        km4.metric("30-Day Trend", f"{ftp:+.1f}%" if pd.notna(ftp) else "—")
    except Exception:
        km4.metric("30-Day Trend", "—")

    if not fcast.empty:
        # Endemic badge
        try:
            endemic = cr.ENDEMIC_FLAG if 'ENDEMIC_FLAG' in risk.columns else None
            if pd.notna(endemic) and endemic:
                st.warning("Low Forecast Confidence — this country is in sustained endemic decline.")
        except Exception:
            pass

        fig = go.Figure()
        # Historical observed data (before forecast)
        fig.add_trace(go.Scatter(x=history.DATE, y=history.VAL, mode='lines',
                                 name='Historical', line=dict(color=CHART_BLUE, width=2),
                                 fill='tozeroy', fillcolor='rgba(37,99,235,.05)'))
        # 30-day forecast
        fig.add_trace(go.Scatter(x=fcast.DATE, y=fcast.VAL, mode='lines',
                                 name='Forecast', line=dict(color=CHART_RED, width=2, dash='dash')))
        # Actual ground truth during forecast period (validation)
        if not validation.empty:
            fig.add_trace(go.Scatter(x=validation.DATE, y=validation.VAL, mode='lines',
                                     name='Actual (Validation)',
                                     line=dict(color=CHART_GREEN, width=2, dash='dot')))

        # Confidence intervals from Snowflake ML
        if 'UPPER_BOUND' in fcast.columns and 'LOWER_BOUND' in fcast.columns:
            fc_ci = fcast.dropna(subset=['UPPER_BOUND', 'LOWER_BOUND']).copy()
            if not fc_ci.empty:
                fc_ci['UPPER_BOUND'] = fc_ci['UPPER_BOUND'].clip(lower=0)
                fc_ci['LOWER_BOUND'] = fc_ci['LOWER_BOUND'].clip(lower=0)
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc_ci.DATE, fc_ci.DATE[::-1]]),
                    y=pd.concat([fc_ci.UPPER_BOUND, fc_ci.LOWER_BOUND[::-1]]),
                    fill='toself', fillcolor='rgba(220,38,38,.08)',
                    line=dict(color='rgba(0,0,0,0)'), name='95% CI'))

        # Vertical line marking forecast start (must be string for Plotly compat)
        fig.add_vline(x=str(forecast_start)[:10], line_dash='dot', line_color=GRAY_500,
                       annotation_text='Forecast Start', annotation_font_size=11)

        plotly_layout(fig, 420)
        fig.update_layout(yaxis_title='7-Day Avg Cases', yaxis=dict(rangemode='tozero'))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Model trained on Jan 2020 – Mar 2022. Forecast predicts the next 30 days. "
                  "Green line shows what actually happened for validation.")
    else:
        st.info("No forecast data available. Run 03_ml_models.sql first.")

    # Model metrics — pivoted: each country as a row, metrics as columns
    if not metrics_df.empty:
        st.markdown('<div class="sub-hdr">Model Evaluation Metrics</div>', True)
        mdf = metrics_df.copy()
        mdf['SERIES'] = mdf['SERIES'].str.strip('"')
        try:
            pivot = mdf.pivot_table(index='SERIES', columns='ERROR_METRIC',
                                    values='METRIC_VALUE', aggfunc='first')
            pivot = pivot.reset_index().rename(columns={'SERIES': 'Country'})
            metric_order = ['Country']
            for mc in ['MAE', 'SMAPE', 'MDA', 'MSE', 'WINKLER_ALPHA=0.05']:
                if mc in pivot.columns:
                    metric_order.append(mc)
            metric_order += [c for c in pivot.columns if c not in metric_order]
            pivot = pivot[metric_order]
            st.dataframe(pivot, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ═══════════ TAB 4: INTELLIGENCE ═════════════════════════════
with tabs[3]:
    st.markdown(f'<div class="sec-hdr">Intelligence Briefing — {sel_country}</div>', True)

    # Global triage
    if not global_triage.empty:
        with st.expander("Global Situation Summary", expanded=False):
            st.markdown(str(global_triage.iloc[0].GLOBAL_BRIEF))

    # Country health brief
    if not narratives.empty:
        narr = narratives[narratives.COUNTRY_REGION == sel_country]
        if not narr.empty:
            n = narr.iloc[0]
            st.markdown(f'{tier_badge(n.RISK_TIER)} &nbsp; Score: {n.RISK_SCORE}/100', True)
            st.markdown("#### AI Health Brief")
            st.info(str(n.HEALTH_BRIEF))
    else:
        st.info("Cortex narratives not generated yet. Run 04_risk_and_cortex.sql.")

    # Anomaly explanations
    if not anomaly_expl.empty:
        anom_e = anomaly_expl[anomaly_expl.COUNTRY_REGION == sel_country]
        if 'DEVIATION_PCT' in anom_e.columns:
            anom_e = anom_e.sort_values('DEVIATION_PCT', ascending=False, key=abs)
        if not anom_e.empty:
            st.markdown('<div class="sub-hdr">Anomaly Explanations</div>', True)
            for idx, ae in anom_e.iterrows():
                date_str = str(ae.DATE.date()) if hasattr(ae.DATE, 'date') else str(ae.DATE)
                with st.expander(f"{date_str} — {ae.ANOMALY_DIRECTION} ({ae.DEVIATION_PCT:+.0f}%)"):
                    st.markdown(f"**Actual**: {ae.ACTUAL:,.0f} | **Expected**: {ae.EXPECTED:,.0f}")
                    st.markdown(str(ae.EXPLANATION))

    # Live Q&A
    st.markdown('<div class="sub-hdr">Ask Cortex</div>', True)
    suggestions = [
        "Compare to Europe?",
        "Travel restrictions?",
        "Press conference summary?",
    ]
    cols = st.columns(len(suggestions))
    for i, q in enumerate(suggestions):
        if cols[i].button(q, key=f"sugg_{i}"):
            st.session_state["prefill_q"] = q

    user_q = st.text_input("Your question:", value=st.session_state.get("prefill_q", ""))

    if st.button("Ask Cortex") and user_q:
        rt_v = safe_val(cr, 'RT_EFFECTIVE')
        cases_v = safe_val(cr, 'ROLLING_AVG_7D_CASES', "{:,.0f}")
        prompt = f"""You are a public health analyst. COVID-19 data for {sel_country}:
7-day avg cases: {cases_v}. Rt: {rt_v}.
Risk: {cr.RISK_TIER} ({score_val}/100). Phase: {cr.EPIDEMIC_PHASE}.
Answer in under 4 sentences for a non-technical official. Question: {user_q}"""

        try:
            with st.spinner("Cortex is thinking..."):
                result = session.sql(
                    f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', $${prompt}$$)"
                ).collect()[0][0]
            st.info(str(result))
        except Exception as e:
            st.error(f"Cortex error: {e}")

    # All country summaries
    if not narratives.empty:
        st.markdown('<div class="sub-hdr">All Country Briefs</div>', True)
        for idx, n in narratives.iterrows():
            with st.expander(f"{n.COUNTRY_REGION} — {n.RISK_TIER} ({n.RISK_SCORE}/100)"):
                st.markdown(str(n.HEALTH_BRIEF))


# ═══════════ TAB 5: COMPARISON ═══════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-hdr">Comparative Analysis</div>', True)
    default_comp = [sel_country] + [c for c in countries[:4] if c != sel_country][:2]
    comp = st.multiselect("Select countries", countries, default=default_comp, max_selections=5)

    if len(comp) >= 2:
        DATE_START_COMP = pd.Timestamp('2020-06-01')
        comp_data = features[(features.COUNTRY_REGION.isin(comp)) &
                             (features.DATE >= DATE_START_COMP)]

        if not comp_data.empty and 'ROLLING_AVG_7D_CASES' in comp_data.columns:
            st.markdown('<div class="sub-hdr">Case Trends</div>', True)
            fig_comp = go.Figure()
            for country in comp:
                cd = comp_data[comp_data.COUNTRY_REGION == country]
                if not cd.empty:
                    fig_comp.add_trace(go.Scatter(x=cd.DATE, y=cd.ROLLING_AVG_7D_CASES,
                                                  mode='lines', name=country))
            fig_comp.update_layout(yaxis_title='7-Day Avg Cases')
            plotly_layout(fig_comp, 380)
            st.plotly_chart(fig_comp, use_container_width=True)

        rt_data = comp_data.dropna(subset=['RT_EFFECTIVE']) if 'RT_EFFECTIVE' in comp_data.columns else pd.DataFrame()
        if not rt_data.empty:
            st.markdown('<div class="sub-hdr">Reproduction Number</div>', True)
            fig_rt = go.Figure()
            for country in comp:
                rd = rt_data[rt_data.COUNTRY_REGION == country]
                if not rd.empty:
                    fig_rt.add_trace(go.Scatter(x=rd.DATE, y=rd.RT_EFFECTIVE,
                                                mode='lines', name=country))
            fig_rt.add_hline(y=1.0, line_dash='dash', line_color=CHART_RED)
            fig_rt.update_layout(yaxis_title='Rt')
            plotly_layout(fig_rt, 340)
            st.plotly_chart(fig_rt, use_container_width=True)

        comp_risk = risk[risk.COUNTRY_REGION.isin(comp)]
        comp_cols = [c for c in ['COUNTRY_REGION', 'RISK_TIER', 'RISK_SCORE', 'RT_EFFECTIVE',
                                  'EPIDEMIC_PHASE', 'FORECAST_TREND_PCT'] if c in comp_risk.columns]
        comp_tbl = comp_risk[comp_cols].copy().fillna("—")
        st.dataframe(comp_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 countries to compare.")


# ═══════════ TAB 6: ANOMALIES ════════════════════════════════
with tabs[5]:
    st.markdown(f'<div class="sec-hdr">Anomaly Detection — {sel_country}</div>', True)

    c_anom = anomalies[anomalies.COUNTRY_REGION == sel_country].sort_values('DATE')
    c_flag = c_anom[c_anom.IS_ANOMALY == True] if 'IS_ANOMALY' in c_anom.columns else pd.DataFrame()
    all_flagged = anomalies[anomalies.IS_ANOMALY == True] if 'IS_ANOMALY' in anomalies.columns else pd.DataFrame()

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Anomalies ({sel_country})", f"{len(c_flag):,}")
    m2.metric("Total (All Countries)", f"{len(all_flagged):,}")
    m3.metric("Avg Deviation",
              f"{c_flag.DEVIATION_PCT.abs().mean():.0f}%" if (not c_flag.empty and 'DEVIATION_PCT' in c_flag.columns) else "—")

    if not c_anom.empty:
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=c_anom.DATE, y=c_anom.DAILY_NEW_CASES,
                                      mode='lines', name='Observed',
                                      line=dict(color=CHART_BLUE, width=1)))
        if 'EXPECTED' in c_anom.columns:
            fig_anom.add_trace(go.Scatter(x=c_anom.DATE, y=c_anom.EXPECTED,
                                          mode='lines', name='Expected',
                                          line=dict(color='#6B7280', dash='dash')))
        if not c_flag.empty:
            spikes = c_flag[c_flag.ANOMALY_DIRECTION == 'SPIKE'] if 'ANOMALY_DIRECTION' in c_flag.columns else pd.DataFrame()
            drops = c_flag[c_flag.ANOMALY_DIRECTION == 'DROP'] if 'ANOMALY_DIRECTION' in c_flag.columns else pd.DataFrame()
            if not spikes.empty:
                fig_anom.add_trace(go.Scatter(x=spikes.DATE, y=spikes.DAILY_NEW_CASES,
                                              mode='markers', name='Spike',
                                              marker=dict(color=CHART_RED, size=7, symbol='triangle-up')))
            if not drops.empty:
                fig_anom.add_trace(go.Scatter(x=drops.DATE, y=drops.DAILY_NEW_CASES,
                                              mode='markers', name='Drop',
                                              marker=dict(color=CHART_GREEN, size=7, symbol='triangle-down')))
        plotly_layout(fig_anom, 380)
        fig_anom.update_layout(yaxis_title='Daily Cases')
        st.plotly_chart(fig_anom, use_container_width=True)

        if not c_flag.empty and 'DEVIATION_PCT' in c_flag.columns:
            st.markdown(f'<div class="sub-hdr">Top Anomalies — {sel_country}</div>', True)
            top_cols = [c for c in ['DATE', 'DAILY_NEW_CASES', 'EXPECTED', 'DEVIATION_PCT', 'ANOMALY_DIRECTION']
                        if c in c_flag.columns]
            top = c_flag.nlargest(15, 'DEVIATION_PCT', keep='first')[top_cols].copy()
            st.dataframe(top, use_container_width=True, hide_index=True)


# ═══════════ TAB 7: METHODOLOGY ══════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec-hdr">Methodology and Fairness</div>', True)
    st.markdown(f"""
### Data Source
Johns Hopkins University CSSE COVID-19 Dataset via Snowflake Marketplace (Starschema).
**{len(countries)} countries** across all 6 WHO regions. January 2020 — March 2023.

### Tables Used
| Table | Source | Purpose |
|-------|--------|---------|
| `JHU_COVID_19` | JHU CSSE | Case and death time-series |
| `GOOG_GLOBAL_MOBILITY_REPORT` | Google | Mobility index (risk factor) |
| `OWID_VACCINATIONS` | OWID | Vaccination coverage |

### Machine Learning
| Model | Engine | Train Period | Confidence |
|-------|--------|-------------|------------|
| Forecast | SNOWFLAKE.ML.FORECAST | Jan 2020 – Mar 2022 | 95% CI from model |
| Anomaly | SQL Z-Score / SNOWFLAKE.ML | Full series | Statistical threshold |
| Risk | SQL rules (8-factor) | Latest snapshot | Deterministic |

### Train/Test Split
**Fixed calendar split**: Train on Jan 2020 – Mar 31, 2022. Validate on Apr 2022 – Mar 2023.
Every country has an identical validation window for fair MAPE comparison.

### Risk Scoring (8 Factors — All Absolute Thresholds)
| Factor | Max Pts | Threshold |
|--------|---------|-----------|
| Rt value | 25 | >1.5 → 25, >1.0 → 15, >0.8 → 5 |
| Forecast trend | 20 | >50% increase → 20, >10% → 10 |
| Acceleration | 15 | Positive + accelerating → 15 |
| Doubling time | 15 | <14 days → 15, <30 → 8 |
| Week-over-week | 10 | >50% → 10, >20% → 5 |
| CFR | 10 | >3% → 10, >1.5% → 5 |
| Case volume | 5 | >50K/day → 5, >10K → 2 |
| Recent anomaly | 5 | Anomaly in last 7 days → 5 |

### Fairness
- **Africa bias**: Nigeria/South Africa have highest null rates — flagged explicitly
- **Vaccination gap**: OWID data loaded where available; absence documented
- **Endemic tail**: Countries in sustained decline receive low-confidence badge
- **Reporting lag**: 7-day rolling averages correct for weekend effects
- **High-income bias**: MAPE shown per country — wealthier nations have more consistent data
    """)


# ── Footer ───────────────────────────────────────────────────
st.markdown(f'<div class="footer">Sentinel — Epidemiological Intelligence | '
            f'Snowflake ML + Cortex | For research purposes only</div>', True)
