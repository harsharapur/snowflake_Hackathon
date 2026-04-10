"""
Sentinel — COVID-19 Insights for Everyone
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
/* ── Chatbot Section ── */
.chat-section-header {{ background: {MAROON}; color: {WHITE}; padding: 14px 18px;
    border-radius: 8px 8px 0 0; font-weight: 700; font-size: 1rem;
    display: flex; justify-content: space-between; align-items: center; }}
.chat-section-header small {{ font-weight: 400; font-size: .75rem; opacity: .7; }}
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
    try:
        policy_sims = query("SELECT * FROM PUBLIC_HEALTH_DB.ANALYTICS.POLICY_SIMULATIONS ORDER BY COUNTRY_REGION")
    except Exception:
        policy_sims = pd.DataFrame()

    return {
        'risk': risk, 'features': features, 'hf': hf, 'forecast': forecast,
        'anomalies': anomalies, 'narratives': narratives,
        'anomaly_expl': anomaly_expl, 'global_triage': global_triage,
        'vaccinations': vaccinations, 'metrics': metrics_df,
        'policy_sims': policy_sims
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
policy_sims = data['policy_sims']

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


# ── Chatbot Engine ────────────────────────────────────────────
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

TABLE_SCHEMAS = """Available tables (ONLY generate SELECT queries):
1. PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
   Columns: COUNTRY_REGION, DATE, DAILY_NEW_CASES, CUMULATIVE_CONFIRMED, CUMULATIVE_DEATHS,
   ROLLING_AVG_7D_CASES, ROLLING_AVG_14D_CASES, ROLLING_AVG_7D_DEATHS,
   RT_EFFECTIVE, DOUBLING_TIME_DAYS, CASE_FATALITY_RATE, ACCELERATION,
   WOW_CHANGE_PCT, EPIDEMIC_PHASE, TREND_DIRECTION, ENDEMIC_FLAG

2. PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS
   Columns: COUNTRY_REGION, RISK_TIER, RISK_SCORE, RT_EFFECTIVE,
   ROLLING_AVG_7D_CASES, FORECAST_TREND_PCT,
   F1_RT, F2_FORECAST, F3_ACCEL, F4_DOUBLING, F5_WOW, F6_CFR, F7_VOLUME, F8_ANOMALY

3. PUBLIC_HEALTH_DB.ML.FORECASTS
   Columns: COUNTRY_REGION, DATE, FORECASTED_CASES, FORECAST_LOWER, FORECAST_UPPER

4. PUBLIC_HEALTH_DB.ML.ANOMALIES
   Columns: COUNTRY_REGION, DATE, DAILY_NEW_CASES, EXPECTED, DEVIATION_PCT,
   IS_ANOMALY (BOOLEAN), ANOMALY_DIRECTION

5. PUBLIC_HEALTH_DB.ML.FORECAST_METRICS
   Columns: SERIES (=country name), METRIC ('MAPE'/'MAE'), VALUE

6. PUBLIC_HEALTH_DB.ANALYTICS.VW_CORTEX_CONTEXT
   One row per country with ALL combined metrics, forecasts, risk, anomalies."""

def build_screen_context(country_row, country_name, score):
    """Build rich context describing what the user sees on the dashboard."""
    rt_v = safe_val(country_row, 'RT_EFFECTIVE')
    cases_v = safe_val(country_row, 'ROLLING_AVG_7D_CASES', '{:,.0f}')
    phase_v = getattr(country_row, 'EPIDEMIC_PHASE', 'unknown')
    trend_v = getattr(country_row, 'TREND_DIRECTION', 'unknown')
    fcast_v = safe_val(country_row, 'FORECAST_TREND_PCT', '{:.1f}')
    cfr_v = safe_val(country_row, 'CASE_FATALITY_RATE', '{:.2f}')
    tier = country_row.RISK_TIER

    return f"""=== DASHBOARD: Sentinel — COVID-19 Insights for Everyone ===
The user is viewing a COVID-19 intelligence dashboard tracking {len(countries)} countries.
Currently selected country: {country_name}

SIDEBAR (always visible):
- Country selector: {country_name} selected
- Risk card: {tier} risk, score {score}/100

HEADER: Shows "{country_name}" with a {tier} risk badge.

TAB 1 — OVERVIEW (Global Risk):
- KPI cards: {len(risk[risk.RISK_TIER == 'HIGH'])} HIGH risk countries, {len(risk[risk.RISK_TIER == 'MODERATE'])} MODERATE, {len(risk[risk.RISK_TIER == 'LOW'])} LOW
- Average Rt across all countries: {risk.RT_EFFECTIVE.dropna().mean():.2f}
- Average risk score: {risk.RISK_SCORE.mean():.0f}/100
- Bar chart: risk scores for all countries (sorted highest to lowest)
- Map: color-coded by risk tier (red=HIGH, amber=MODERATE, green=LOW)
- Global risk table with all countries

TAB 2 — EPIDEMIOLOGY (for {country_name}):
- KPI cards: Rt={rt_v}, 7d avg cases={cases_v}, Phase={phase_v}, CFR={cfr_v}%
- Chart: 7-day rolling average cases over time (Jan 2020 – Mar 2023)
- Chart: Rt (reproduction number) over time with threshold line at 1.0
- Chart: Daily deaths rolling average

TAB 3 — FORECAST (for {country_name}):
- ML forecast chart: 30-day prediction with 95% confidence interval (shaded band)
- Historical actual cases vs predicted (model training on 2020–2022, validation 2022–2023)
- Forecast trend: {fcast_v}% change over 30 days
- Model performance table: MAPE and MAE per country

TAB 4 — INTELLIGENCE (for {country_name}):
- Quick Summary: AI-generated overview paragraph
- What Do These Numbers Mean?: AI explains each metric with analogies
- What's the Situation?: 4-paragraph deep-dive analysis
- What Can I Do to Stay Safe?: 5 data-driven safety recommendations
- Compare Countries: user selects multiple countries for AI comparison
- What Could Happen Next?: 3 scenario simulations (status quo, new variant, intervention)
- Anomaly Explanations: AI explains detected unusual data patterns

TAB 5 — COMPARISON:
- Side-by-side charts for multiple countries (case trends, Rt)
- Comparison table with risk scores

TAB 6 — ANOMALIES (for {country_name}):
- Chart: actual vs expected cases with anomaly flags
- List of detected anomalies with deviation percentages
- Top anomalies table

TAB 7 — METHODOLOGY:
- Data sources, ML model details, risk scoring formula, AI grounding strategy

CURRENT DATA FOR {country_name.upper()}:
- Risk: {tier} ({score}/100)
- Rt: {rt_v} (each infected person spreads to this many others)
- 7-day avg cases: {cases_v}/day
- Epidemic phase: {phase_v}
- Trend direction: {trend_v}
- Forecast trend: {fcast_v}% over next 30 days
- Case fatality rate: {cfr_v}%
- Risk breakdown: Rt={safe_val(country_row, 'F1_RT', '{:.0f}')}/25, Forecast={safe_val(country_row, 'F2_FORECAST', '{:.0f}')}/20, Accel={safe_val(country_row, 'F3_ACCEL', '{:.0f}')}/15, Doubling={safe_val(country_row, 'F4_DOUBLING', '{:.0f}')}/15, WoW={safe_val(country_row, 'F5_WOW', '{:.0f}')}/10, CFR={safe_val(country_row, 'F6_CFR', '{:.0f}')}/10, Volume={safe_val(country_row, 'F7_VOLUME', '{:.0f}')}/5, Anomaly={safe_val(country_row, 'F8_ANOMALY', '{:.0f}')}/5"""


def extract_sql(response_text):
    """Extract SQL query from Cortex response if present. Returns None if no query."""
    text = str(response_text)
    if '```sql' in text:
        sql = text.split('```sql')[1].split('```')[0].strip()
    elif '```' in text and 'SELECT' in text.upper():
        sql = text.split('```')[1].split('```')[0].strip()
    else:
        return None
    upper = sql.upper().strip()
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
                 'TRUNCATE', 'MERGE', 'GRANT', 'REVOKE', 'EXEC']
    if not upper.startswith('SELECT'):
        return None
    if any(kw in upper for kw in forbidden):
        return None
    return sql

def chat_respond(user_message, country_row, country_name, score):
    """Process user message through 2-step Cortex pipeline."""
    screen_ctx = build_screen_context(country_row, country_name, score)

    # Build recent conversation history (last 6 messages)
    history = ""
    for msg in st.session_state.chat_messages[-6:]:
        role_label = "User" if msg['role'] == 'user' else "Sentinel"
        history += f"{role_label}: {msg['content']}\n"

    system = (
        "You are Sentinel, a COVID-19 data assistant built into an epidemiological dashboard. "
        "You are an expert analyst but you write for everyday people with no medical background.\n\n"
        "STRICT RULES:\n"
        "1. ONLY answer questions about COVID-19 data, epidemiology, public health, and what the dashboard shows.\n"
        "2. If asked about ANYTHING else (weather, sports, coding, recipes, politics, etc.), politely say: "
        "'I can only help with COVID-19 related questions. Try asking about case trends, risk levels, or forecasts!'\n"
        "3. NEVER generate INSERT, UPDATE, DELETE, DROP, or any modifying SQL.\n"
        "4. If you need to look up data, write a single SELECT query wrapped in ```sql ... ```\n"
        "5. If you can answer from the screen context without a query, just answer directly.\n"
        "6. Always cite specific numbers. Explain what they mean in simple terms.\n"
        "7. Keep answers concise — under 150 words unless the user asks for detail.\n\n"
        f"{TABLE_SCHEMAS}\n\n"
        f"CURRENT SCREEN CONTEXT:\n{screen_ctx}\n\n"
        f"CONVERSATION HISTORY:\n{history}\n"
    )

    step1_prompt = f"{system}User question: {user_message}"

    try:
        response = session.sql(
            f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', $${step1_prompt}$$)"
        ).collect()[0][0]
    except Exception as e:
        return f"Sorry, I had trouble processing that. Please try again. ({str(e)[:80]})"

    sql_query = extract_sql(response)

    if sql_query:
        try:
            query_result = session.sql(sql_query).to_pandas()
            result_str = query_result.to_string(index=False, max_rows=20)

            step2_prompt = (
                f"You are Sentinel, a COVID-19 data assistant for everyday people.\n\n"
                f"The user asked: {user_message}\n\n"
                f"You queried the database and got these results:\n{result_str}\n\n"
                f"Current context: viewing {country_name} ({country_row.RISK_TIER} risk, score {score}/100)\n\n"
                f"Explain the results in simple, plain English anyone can understand. "
                f"Reference the specific numbers. Be concise (under 150 words). "
                f"If the results are empty, say so and suggest what to ask instead."
            )
            explanation = session.sql(
                f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', $${step2_prompt}$$)"
            ).collect()[0][0]
            return str(explanation)
        except Exception as e:
            # Query failed — fall back to the raw response without SQL
            clean = str(response)
            if '```' in clean:
                clean = clean.split('```')[0].strip()
            return clean if clean else f"Could you rephrase? I had trouble looking that up. ({str(e)[:60]})"
    else:
        return str(response)


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
    st.markdown(f'<div class="sec-hdr">COVID-19 Insights — {sel_country}</div>', True)

    # ── 1. Global Situation ──
    if not global_triage.empty:
        with st.expander("🌍 What’s Happening Around the World", expanded=False):
            st.markdown(str(global_triage.iloc[0].GLOBAL_BRIEF))

    # Get country narrative row
    narr_row = None
    if not narratives.empty:
        narr = narratives[narratives.COUNTRY_REGION == sel_country]
        if not narr.empty:
            narr_row = narr.iloc[0]

    if narr_row is not None:
        st.markdown(f'{tier_badge(narr_row.RISK_TIER)} &nbsp; Score: {int(narr_row.RISK_SCORE)}/100', True)

        # ── 2. Executive Summary ──
        if hasattr(narr_row, 'EXECUTIVE_BRIEF') and pd.notna(narr_row.EXECUTIVE_BRIEF):
            with st.expander("📋 Quick Summary", expanded=True):
                st.markdown(str(narr_row.EXECUTIVE_BRIEF))

        # ── 3. What Do These Numbers Mean? ──
        if hasattr(narr_row, 'METRIC_EXPLAINER') and pd.notna(narr_row.METRIC_EXPLAINER):
            with st.expander("🔢 What Do These Numbers Mean?", expanded=False):
                st.markdown(str(narr_row.METRIC_EXPLAINER))

        # ── 4. Situation Report ──
        if hasattr(narr_row, 'SITUATION_SUMMARY') and pd.notna(narr_row.SITUATION_SUMMARY):
            with st.expander("📖 What’s the Situation?", expanded=False):
                st.markdown(str(narr_row.SITUATION_SUMMARY))

        # ── 5. Preventive Measures ──
        if hasattr(narr_row, 'PREVENTIVE_MEASURES') and pd.notna(narr_row.PREVENTIVE_MEASURES):
            with st.expander("🛡️ What Can I Do to Stay Safe?", expanded=False):
                st.markdown(str(narr_row.PREVENTIVE_MEASURES))
    else:
        st.info("Cortex narratives not generated yet. Run 04_risk_and_cortex.sql.")

    # ── 6. Compare Countries ──
    st.markdown('<div class="sub-hdr">⚖️ Compare Countries</div>', True)
    compare_countries = st.multiselect(
        "Select countries to compare",
        countries,
        default=[],
        max_selections=4,
        key="intel_compare"
    )

    if len(compare_countries) >= 2 and st.button("🔍 Generate Comparison", key="compare_btn"):
        # Build context for each selected country from VW_CORTEX_CONTEXT
        country_data_parts = []
        for cc in compare_countries:
            cc_risk = risk[risk.COUNTRY_REGION == cc]
            if cc_risk.empty:
                continue
            ccr = cc_risk.iloc[0]
            country_data_parts.append(
                f"{cc}: Rt={safe_val(ccr, 'RT_EFFECTIVE')}, "
                f"7d avg={safe_val(ccr, 'ROLLING_AVG_7D_CASES', '{:,.0f}')}, "
                f"Phase={getattr(ccr, 'EPIDEMIC_PHASE', '—')}, "
                f"Risk={getattr(ccr, 'RISK_TIER', '—')} ({safe_val(ccr, 'RISK_SCORE', '{:.0f}')}/100), "
                f"Forecast trend={safe_val(ccr, 'FORECAST_TREND_PCT', '{:.1f}')}%"
            )

        countries_str = " | ".join(country_data_parts)
        prompt = f"""You are a senior epidemiologist comparing COVID-19 situations across countries.
Your audience is regular people with no medical background. Do expert-level analysis but explain everything simply.

Data:
{countries_str}

Write a structured comparison covering:
1. Which country is dealing with the toughest situation and why (cite the numbers)
2. How the epidemic trajectories differ across these countries
3. What people in each country can learn from the others
4. One practical takeaway for each country

Reference the specific numbers and explain what they mean. Analyze deeply, write simply.
Under 400 words. No bullet points."""

        try:
            with st.spinner("Cortex is comparing..."):
                result = session.sql(
                    f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', $${prompt}$$)"
                ).collect()[0][0]
            st.info(str(result))
        except Exception as e:
            st.error(f"Cortex comparison error: {e}")
    elif len(compare_countries) == 1:
        st.caption("Select at least 2 countries to compare.")

    # ── 7. Policy Simulation ──
    if not policy_sims.empty:
        sim_row = policy_sims[policy_sims.COUNTRY_REGION == sel_country]
        if not sim_row.empty:
            with st.expander("🔮 What Could Happen Next? — Different Scenarios", expanded=False):
                st.markdown(str(sim_row.iloc[0].SCENARIO_ANALYSIS))

    # ── 8. Anomaly Explanations ──
    if not anomaly_expl.empty:
        anom_e = anomaly_expl[anomaly_expl.COUNTRY_REGION == sel_country]
        if 'DEVIATION_PCT' in anom_e.columns:
            anom_e = anom_e.sort_values('DEVIATION_PCT', ascending=False, key=abs)
        if not anom_e.empty:
            st.markdown('<div class="sub-hdr">🚨 Anomaly Explanations</div>', True)
            for idx, ae in anom_e.iterrows():
                date_str = str(ae.DATE.date()) if hasattr(ae.DATE, 'date') else str(ae.DATE)
                with st.expander(f"{date_str} — {ae.ANOMALY_DIRECTION} ({ae.DEVIATION_PCT:+.0f}%)"):
                    st.markdown(f"**Actual**: {ae.ACTUAL:,.0f} | **Expected**: {ae.EXPECTED:,.0f}")
                    st.markdown(str(ae.EXPLANATION))


    # ── All Country Summaries ──
    if not narratives.empty:
        col_check = 'EXECUTIVE_BRIEF' if 'EXECUTIVE_BRIEF' in narratives.columns else ('SITUATION_SUMMARY' if 'SITUATION_SUMMARY' in narratives.columns else None)
        if col_check:
            st.markdown('<div class="sub-hdr">All Country Summaries</div>', True)
            for idx, n in narratives.iterrows():
                brief_text = str(getattr(n, col_check, ''))
                if brief_text and brief_text != 'nan':
                    with st.expander(f"{n.COUNTRY_REGION} — {n.RISK_TIER} ({int(n.RISK_SCORE)}/100)"):
                        st.markdown(brief_text)


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
| `VW_CORTEX_CONTEXT` | Pipeline View | One-row-per-country context for AI grounding |
| `CORTEX_NARRATIVES` | Cortex AI | 4-column AI intelligence per country |
| `POLICY_SIMULATIONS` | Cortex AI | 3-scenario what-if analysis per country |

### Machine Learning
| Model | Engine | Train Period | Confidence |
|-------|--------|-------------|------------|
| Forecast | SNOWFLAKE.ML.FORECAST | Jan 2020 – Mar 2022 | 95% CI from model |
| Anomaly | SQL Z-Score / SNOWFLAKE.ML | Full series | Statistical threshold |
| Risk | SQL rules (8-factor) | Latest snapshot | Deterministic |

### Train/Test Split
**Fixed calendar split**: Train on Jan 2020 – Mar 31, 2022. Validate on Apr 2022 – Mar 2023.
Every country has an identical validation window for fair MAPE comparison.

### AI Intelligence (Cortex)
| Narrative | Model | Grounding |
|-----------|-------|-----------|
| Metric Explainer | mistral-large | All 25+ features + 8-factor risk breakdown + ML forecast trajectory + MAPE |
| Situation Summary | mistral-large | Current data + forecast + anomalies + historical wave context |
| Preventive Measures | mistral-large | Risk level + epidemic phase + forecast + model confidence |
| Policy Simulation | mistral-large | ML predictions + historical peaks + risk factors |
| Compare Countries | mistral-large (live) | Risk tiers + Rt + forecast trends for selected countries |
| Sentinel Chatbot | mistral-large (live) | 2-step: SQL generation + plain English explanation |

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

### AI Grounding Strategy
All AI narratives are grounded in **actual pipeline data** — not external documents.
Every claim references specific numbers the user can verify in the dashboard:
- Epidemiological indicators from `COVID_FEATURES`
- 8-factor risk breakdown from `RISK_TIERS`
- 30-day forecast trajectory + MAPE from `FORECASTS` / `FORECAST_METRICS`
- Anomaly detection stats from `ANOMALIES`
- Historical wave peaks from computed analysis

### Fairness
- **Africa bias**: Nigeria/South Africa have highest null rates — flagged explicitly
- **Vaccination gap**: OWID data loaded where available; absence documented
- **Endemic tail**: Countries in sustained decline receive low-confidence badge
- **Reporting lag**: 7-day rolling averages correct for weekend effects
- **High-income bias**: MAPE shown per country — wealthier nations have more consistent data
    """)


# ── Chatbot ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div class="chat-section-header">
    <span>Ask Sentinel</span>
    <small>{sel_country} | {cr.RISK_TIER}</small>
</div>
""", True)

# Suggested quick questions
sugg_questions = [
    "Explain my country's risk score",
    "What does the forecast predict?",
    "What does Rt mean for me?",
    "Any unusual patterns detected?",
    "How is the world doing overall?",
    "What should I do to stay safe?",
]
sugg_cols = st.columns(3)
for i, sq in enumerate(sugg_questions):
    if sugg_cols[i % 3].button(sq, key=f"chat_sq_{i}"):
        st.session_state.chat_pending = sq
        st.rerun()

# Display chat history
if not st.session_state.chat_messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"Hi! I'm **Sentinel**, your COVID-19 data assistant. "
            f"I can help you understand anything about COVID-19 data for **{sel_country}** "
            f"or any other country in the dashboard. Ask me anything!"
        )
else:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
chat_input = st.chat_input("Ask me anything about COVID-19 data...")

# Process pending suggestion or typed input
pending = st.session_state.pop("chat_pending", None)
if chat_input:
    pending = chat_input

if pending:
    st.session_state.chat_messages.append({"role": "user", "content": pending})
    with st.chat_message("user"):
        st.markdown(pending)
    with st.chat_message("assistant"):
        with st.spinner("Looking up the data..."):
            response = chat_respond(pending, cr, sel_country, score_val)
        st.markdown(response)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()

# Clear chat button
if st.session_state.chat_messages:
    if st.button("Clear chat", key="chat_clear"):
        st.session_state.chat_messages = []
        st.rerun()


# ── Footer ───────────────────────────────────────────────────
st.markdown(f'<div class="footer">Sentinel — COVID-19 Insights for Everyone | '
            f'Snowflake ML + Cortex | For research purposes only</div>', True)
