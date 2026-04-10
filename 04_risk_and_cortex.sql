-- ============================================================================
-- SENTINEL — PUBLIC HEALTH INTELLIGENCE PLATFORM
-- FILE 4 OF 4: RISK TIERS + CORTEX AI NARRATIVES
-- ============================================================================
-- Run after 03_ml_models.sql
-- Time: ~5 minutes
--
-- FIX: latest CTE now uses QUALIFY ROW_NUMBER() per country — not global
--      MAX(DATE). Countries that stopped reporting early now appear.
-- FIX: Factor 2 is NULL-safe when forecast is missing.
-- ============================================================================

USE ROLE TRAINING_ROLE;
USE WAREHOUSE SYSTEM$STREAMLIT_NOTEBOOK_WH;
USE DATABASE PUBLIC_HEALTH_DB;
USE SCHEMA ANALYTICS;

-- ╔════════════════════════════════════════════════════════════════╗
-- ║  4.1  RISK TIER CLASSIFICATION — 8-FACTOR COMPOSITE SCORE     ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS AS
WITH latest AS (
    -- Most recent row PER COUNTRY using QUALIFY.
    -- FIX: Previously used WHERE DATE = MAX(DATE) globally — countries
    --      that stopped reporting before the last date had zero rows,
    --      causing them to be silently dropped from RISK_TIERS.
    SELECT *
    FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
    QUALIFY ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE DESC) = 1
),
fc_summary AS (
    -- Last and first forecast values, clamped to >= 0 (cases can't be negative)
    SELECT
        COUNTRY_REGION,
        GREATEST(MAX(CASE WHEN RN_DESC = 1 THEN FORECASTED_CASES END), 0) AS FC_END,
        GREATEST(MAX(CASE WHEN RN_ASC  = 1 THEN FORECASTED_CASES END), 0) AS FC_START,
        GREATEST(MAX(FORECASTED_CASES), 0)                                 AS FC_PEAK,
        GREATEST(AVG(FORECASTED_CASES), 0)                                 AS FC_AVG
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE DESC) AS RN_DESC,
            ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ASC)  AS RN_ASC
        FROM PUBLIC_HEALTH_DB.ML.FORECASTS
    )
    GROUP BY COUNTRY_REGION
),
recent_anomalies AS (
    -- Were there anomalies in the last 7 days of data for each country?
    SELECT
        a.COUNTRY_REGION,
        MAX(CASE WHEN a.IS_ANOMALY THEN 1 ELSE 0 END) AS HAS_RECENT_ANOMALY
    FROM PUBLIC_HEALTH_DB.ML.ANOMALIES a
    INNER JOIN (
        SELECT COUNTRY_REGION, MAX(DATE) AS LATEST
        FROM PUBLIC_HEALTH_DB.ML.ANOMALIES
        GROUP BY COUNTRY_REGION
    ) mx ON a.COUNTRY_REGION = mx.COUNTRY_REGION
        AND a.DATE >= DATEADD('day', -7, mx.LATEST)
    GROUP BY a.COUNTRY_REGION
),
mobility_latest AS (
    SELECT COUNTRY_REGION, MAX(MOBILITY_INDEX) AS MOBILITY_INDEX
    FROM PUBLIC_HEALTH_DB.RAW.MOBILITY
    WHERE DATE >= DATEADD('day', -14,
        (SELECT COALESCE(MAX(DATE), CURRENT_DATE()) FROM PUBLIC_HEALTH_DB.RAW.MOBILITY))
    GROUP BY COUNTRY_REGION
),
scored AS (
    SELECT
        l.COUNTRY_REGION,
        l.DATE,
        l.ROLLING_AVG_7D_CASES,
        l.RT_EFFECTIVE,
        l.DOUBLING_TIME_DAYS,
        l.EPIDEMIC_PHASE,
        l.CASE_FATALITY_RATE,
        l.WOW_CHANGE_PCT,
        l.ACCELERATION,
        l.TREND_DIRECTION,
        l.ENDEMIC_FLAG,
        l.SIGNAL_TO_NOISE,
        l.CUMULATIVE_CONFIRMED,
        l.CUMULATIVE_DEATHS,
        f.FC_END,
        f.FC_PEAK,
        -- DATA_GAP flag: TRUE when country stopped reporting (7-day rolling avg = 0 or NULL)
        CASE WHEN COALESCE(l.ROLLING_AVG_7D_CASES, 0) = 0 THEN TRUE ELSE FALSE END AS DATA_GAP,

        -- FORECAST_TREND_PCT: % change day1 → day30 in forecast window. Min -100%.
        GREATEST(
            ROUND(
                CASE WHEN f.FC_START > 0 AND f.FC_END IS NOT NULL AND f.FC_START IS NOT NULL
                     THEN (f.FC_END - f.FC_START) / f.FC_START * 100
                     ELSE 0 END, 1
            ), -100
        ) AS FORECAST_TREND_PCT,

        -- ── 8-FACTOR SCORING (ALL ABSOLUTE THRESHOLDS) ──────────

        -- Factor 1: Rt (25 pts)
        CASE WHEN l.RT_EFFECTIVE > 1.5 THEN 25
             WHEN l.RT_EFFECTIVE > 1.0 THEN 15
             WHEN l.RT_EFFECTIVE > 0.8 THEN 5
             ELSE 0 END AS F1_RT,

        -- Factor 2: Forecast trend within the 30-day window (20 pts)
        -- Uses FC_START → FC_END slope (model's own prediction trajectory),
        -- not vs current actual (which is from a different year)
        CASE WHEN f.FC_START > 0 AND f.FC_END > f.FC_START * 1.5 THEN 20
             WHEN f.FC_START > 0 AND f.FC_END > f.FC_START * 1.1 THEN 10
             ELSE 0 END AS F2_FORECAST,

        -- Factor 3: Acceleration (15 pts)
        CASE WHEN l.ACCELERATION > 0 AND l.TREND_DIRECTION = 'ACCELERATING' THEN 15
             ELSE 0 END AS F3_ACCEL,

        -- Factor 4: Doubling time (15 pts)
        CASE WHEN l.DOUBLING_TIME_DAYS IS NOT NULL AND l.DOUBLING_TIME_DAYS < 14 THEN 15
             WHEN l.DOUBLING_TIME_DAYS IS NOT NULL AND l.DOUBLING_TIME_DAYS < 30 THEN 8
             ELSE 0 END AS F4_DOUBLING,

        -- Factor 5: Week-over-week change (10 pts)
        CASE WHEN l.WOW_CHANGE_PCT > 50 THEN 10
             WHEN l.WOW_CHANGE_PCT > 20 THEN 5
             ELSE 0 END AS F5_WOW,

        -- Factor 6: Case fatality rate (10 pts)
        CASE WHEN l.CASE_FATALITY_RATE > 3 THEN 10
             WHEN l.CASE_FATALITY_RATE > 1.5 THEN 5
             ELSE 0 END AS F6_CFR,

        -- Factor 7: Absolute case volume (5 pts)
        CASE WHEN l.ROLLING_AVG_7D_CASES > 50000 THEN 5
             WHEN l.ROLLING_AVG_7D_CASES > 10000 THEN 2
             ELSE 0 END AS F7_VOLUME,

        -- Factor 8: Recent anomaly (5 pts)
        CASE WHEN COALESCE(a.HAS_RECENT_ANOMALY, 0) = 1 THEN 5
             ELSE 0 END AS F8_ANOMALY,

        COALESCE(m.MOBILITY_INDEX, 0) AS MOBILITY_INDEX

    FROM latest l
    LEFT JOIN fc_summary f       ON l.COUNTRY_REGION = f.COUNTRY_REGION
    LEFT JOIN recent_anomalies a ON l.COUNTRY_REGION = a.COUNTRY_REGION
    LEFT JOIN mobility_latest m  ON l.COUNTRY_REGION = m.COUNTRY_REGION
)
SELECT
    s.*,
    LEAST(F1_RT + F2_FORECAST + F3_ACCEL + F4_DOUBLING + F5_WOW + F6_CFR + F7_VOLUME + F8_ANOMALY, 100)
        AS RISK_SCORE,
    CASE
        WHEN F1_RT + F2_FORECAST + F3_ACCEL + F4_DOUBLING + F5_WOW + F6_CFR + F7_VOLUME + F8_ANOMALY >= 50
            THEN 'HIGH'
        WHEN F1_RT + F2_FORECAST + F3_ACCEL + F4_DOUBLING + F5_WOW + F6_CFR + F7_VOLUME + F8_ANOMALY >= 25
            THEN 'MODERATE'
        ELSE 'LOW'
    END AS RISK_TIER
FROM scored s
ORDER BY RISK_SCORE DESC;

-- Verify — should now show all countries from FEATURES
SELECT COUNT(DISTINCT COUNTRY_REGION) AS COUNTRY_COUNT FROM PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS;

SELECT COUNTRY_REGION, RISK_TIER, RISK_SCORE,
       F1_RT, F2_FORECAST, F3_ACCEL, F4_DOUBLING, F5_WOW, F6_CFR, F7_VOLUME, F8_ANOMALY
FROM PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS
ORDER BY RISK_SCORE DESC;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  4.1B  CONTEXT VIEW FOR CORTEX GROUNDING                      ║
-- ╚════════════════════════════════════════════════════════════════╝
-- One row per country with ALL ML + epidemiological data needed
-- for grounded AI narratives. Joins: features, risk breakdown,
-- forecast trajectory, model evaluation (MAPE), anomaly stats,
-- historical wave context, WHO classification, and government policies.

CREATE OR REPLACE VIEW PUBLIC_HEALTH_DB.ANALYTICS.VW_CORTEX_CONTEXT AS
WITH latest_features AS (
    SELECT * FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE DESC) AS RN
        FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
    ) WHERE RN = 1
),
-- ═══ ML FORECAST DATA — Full 30-day trajectory ═══
forecast_detail AS (
    SELECT
        COUNTRY_REGION,
        MIN(FORECASTED_CASES) AS FORECAST_MIN,
        MAX(FORECASTED_CASES) AS FORECAST_MAX,
        AVG(FORECASTED_CASES) AS FORECAST_AVG,
        MAX(CASE WHEN RN_ASC = 1 THEN FORECASTED_CASES END) AS FORECAST_DAY1,
        MAX(CASE WHEN RN_DESC = 1 THEN FORECASTED_CASES END) AS FORECAST_DAY30,
        MAX(CASE WHEN FORECASTED_CASES = MAX_FC THEN DATE END) AS FORECAST_PEAK_DATE,
        MAX(MAX_FC) AS FORECAST_PEAK_VALUE,
        MIN(FORECAST_LOWER) AS CI_LOWER_MIN,
        MAX(FORECAST_UPPER) AS CI_UPPER_MAX,
        AVG(FORECAST_UPPER - FORECAST_LOWER) AS CI_AVG_WIDTH,
        MAX(CASE WHEN RN_DESC = 1 THEN DATE END) AS FORECAST_END_DATE
    FROM (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ASC) AS RN_ASC,
            ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE DESC) AS RN_DESC,
            MAX(FORECASTED_CASES) OVER (PARTITION BY COUNTRY_REGION) AS MAX_FC
        FROM PUBLIC_HEALTH_DB.ML.FORECASTS
    )
    GROUP BY COUNTRY_REGION
),
-- ═══ ML MODEL QUALITY — MAPE from evaluation metrics ═══
model_quality AS (
    SELECT
        SERIES AS COUNTRY_REGION,
        MAX(CASE WHEN METRIC = 'MAPE' THEN VALUE END) AS MAPE,
        MAX(CASE WHEN METRIC = 'MAE' THEN VALUE END) AS MAE
    FROM PUBLIC_HEALTH_DB.ML.FORECAST_METRICS
    GROUP BY SERIES
),
-- ═══ ANOMALY ANALYSIS — Full stats ═══
anomaly_stats AS (
    SELECT
        COUNTRY_REGION,
        COUNT(CASE WHEN IS_ANOMALY THEN 1 END) AS TOTAL_ANOMALIES,
        COUNT(CASE WHEN IS_ANOMALY AND ANOMALY_DIRECTION = 'SPIKE' THEN 1 END) AS SPIKE_COUNT,
        COUNT(CASE WHEN IS_ANOMALY AND ANOMALY_DIRECTION = 'DROP' THEN 1 END) AS DROP_COUNT,
        MAX(CASE WHEN IS_ANOMALY THEN ABS(DEVIATION_PCT) END) AS MAX_DEVIATION_PCT,
        MAX(CASE WHEN IS_ANOMALY THEN DATE END) AS LAST_ANOMALY_DATE,
        AVG(CASE WHEN IS_ANOMALY THEN ABS(DEVIATION_PCT) END) AS AVG_DEVIATION_PCT
    FROM PUBLIC_HEALTH_DB.ML.ANOMALIES
    GROUP BY COUNTRY_REGION
),
-- ═══ HISTORICAL WAVE CONTEXT — compare current vs all-time peaks ═══
wave_context AS (
    SELECT
        COUNTRY_REGION,
        MAX(ROLLING_AVG_7D_CASES) AS ALL_TIME_PEAK_CASES,
        MAX(CASE WHEN ROLLING_AVG_7D_CASES = pk THEN DATE END) AS PEAK_DATE,
        COUNT(DISTINCT EPIDEMIC_PHASE) AS PHASES_OBSERVED
    FROM (
        SELECT *,
            MAX(ROLLING_AVG_7D_CASES) OVER (PARTITION BY COUNTRY_REGION) AS pk
        FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
    )
    GROUP BY COUNTRY_REGION
)
SELECT
    f.COUNTRY_REGION,
    f.DATE AS DATA_DATE,
    -- ═══ Key metrics ═══
    f.ROLLING_AVG_7D_CASES,
    f.ROLLING_AVG_14D_CASES,
    f.ROLLING_AVG_7D_DEATHS,
    f.DAILY_NEW_CASES,
    f.CUMULATIVE_CONFIRMED,
    f.CUMULATIVE_DEATHS,
    -- ═══ Epidemiological indicators ═══
    f.RT_EFFECTIVE,
    f.DOUBLING_TIME_DAYS,
    f.CASE_FATALITY_RATE,
    f.ACCELERATION,
    f.WOW_CHANGE_PCT,
    f.SIGNAL_TO_NOISE,
    f.VOLATILITY_14D,
    -- ═══ Classifications ═══
    f.EPIDEMIC_PHASE,
    f.TREND_DIRECTION,
    f.ENDEMIC_FLAG,
    -- ═══ 8-Factor Risk Breakdown ═══
    r.RISK_SCORE,
    r.RISK_TIER,
    r.FORECAST_TREND_PCT,
    r.F1_RT, r.F2_FORECAST, r.F3_ACCEL, r.F4_DOUBLING,
    r.F5_WOW, r.F6_CFR, r.F7_VOLUME, r.F8_ANOMALY,
    -- ═══ ML FORECAST — Full trajectory ═══
    fd.FORECAST_DAY1,
    fd.FORECAST_DAY30,
    fd.FORECAST_AVG,
    fd.FORECAST_PEAK_VALUE,
    fd.FORECAST_PEAK_DATE,
    fd.CI_LOWER_MIN,
    fd.CI_UPPER_MAX,
    fd.CI_AVG_WIDTH,
    fd.FORECAST_END_DATE,
    -- ═══ ML MODEL QUALITY ═══
    mq.MAPE AS MODEL_MAPE,
    mq.MAE AS MODEL_MAE,
    CASE WHEN mq.MAPE IS NOT NULL AND mq.MAPE < 10 THEN 'HIGH'
         WHEN mq.MAPE IS NOT NULL AND mq.MAPE < 25 THEN 'MODERATE'
         WHEN mq.MAPE IS NOT NULL THEN 'LOW'
         ELSE 'UNKNOWN' END AS MODEL_CONFIDENCE,
    -- ═══ ANOMALY STATS ═══
    COALESCE(an.TOTAL_ANOMALIES, 0) AS TOTAL_ANOMALIES,
    COALESCE(an.SPIKE_COUNT, 0) AS ANOMALY_SPIKES,
    COALESCE(an.DROP_COUNT, 0) AS ANOMALY_DROPS,
    an.MAX_DEVIATION_PCT,
    an.AVG_DEVIATION_PCT,
    an.LAST_ANOMALY_DATE,
    -- ═══ HISTORICAL WAVE CONTEXT ═══
    wc.ALL_TIME_PEAK_CASES,
    wc.PEAK_DATE AS ALL_TIME_PEAK_DATE,
    ROUND(f.ROLLING_AVG_7D_CASES / NULLIF(wc.ALL_TIME_PEAK_CASES, 0) * 100, 1)
        AS PCT_OF_ALL_TIME_PEAK
FROM latest_features f
LEFT JOIN PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS r
    ON f.COUNTRY_REGION = r.COUNTRY_REGION
LEFT JOIN forecast_detail fd
    ON f.COUNTRY_REGION = fd.COUNTRY_REGION
LEFT JOIN model_quality mq
    ON f.COUNTRY_REGION = mq.COUNTRY_REGION
LEFT JOIN anomaly_stats an
    ON f.COUNTRY_REGION = an.COUNTRY_REGION
LEFT JOIN wave_context wc
    ON f.COUNTRY_REGION = wc.COUNTRY_REGION;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  4.2  CORTEX AI — ENHANCED COUNTRY NARRATIVES                 ║
-- ╚════════════════════════════════════════════════════════════════╝
-- 4 columns: METRIC_EXPLAINER, SITUATION_SUMMARY, PREVENTIVE_MEASURES, EXECUTIVE_BRIEF
-- All grounded in actual data from VW_CORTEX_CONTEXT.
-- Model: mistral-large
-- Time: ~8-12 min (3 Cortex calls × 15 countries)

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ANALYTICS.CORTEX_NARRATIVES AS
SELECT
    c.COUNTRY_REGION, c.RISK_TIER, c.RISK_SCORE,
    c.RT_EFFECTIVE, c.ROLLING_AVG_7D_CASES,
    c.EPIDEMIC_PHASE, c.FORECAST_TREND_PCT,

    -- ═══ COLUMN 1: METRIC EXPLAINER ═══
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large', CONCAT(
        'You are explaining COVID-19 data to a government official with ZERO science background.',
        ' Explain what each of these numbers means for ', c.COUNTRY_REGION,
        ' and whether it is good, concerning, or critical.\n\n',
        'DATA (these are actual measured and ML-predicted values — cite them directly):\n',
        '• Rt (reproduction number) = ', COALESCE(ROUND(c.RT_EFFECTIVE,2)::STRING, 'unavailable'),
        ' — each infected person spreads to this many others\n',
        '• Doubling time = ', COALESCE(ROUND(c.DOUBLING_TIME_DAYS,0)::STRING, 'N/A'),
        ' days — how long for total cases to double\n',
        '• Case Fatality Rate = ', ROUND(c.CASE_FATALITY_RATE,2)::STRING, '%\n',
        '• Epidemic phase = ', c.EPIDEMIC_PHASE, '\n',
        '• 7-day avg cases = ', ROUND(c.ROLLING_AVG_7D_CASES,0)::STRING, '/day\n',
        '• Week-over-week change = ', ROUND(c.WOW_CHANGE_PCT,1)::STRING, '%\n',
        '• Risk score = ', c.RISK_SCORE::STRING, '/100 (',
        'breakdown: Rt=', c.F1_RT::STRING, '/25, Forecast=', c.F2_FORECAST::STRING,
        '/20, Accel=', c.F3_ACCEL::STRING, '/15, Doubling=', c.F4_DOUBLING::STRING,
        '/15, WoW=', c.F5_WOW::STRING, '/10, CFR=', c.F6_CFR::STRING,
        '/10, Volume=', c.F7_VOLUME::STRING, '/5, Anomaly=', c.F8_ANOMALY::STRING, '/5)\n',
        '• ML forecast: cases going from ', COALESCE(ROUND(c.FORECAST_DAY1,0)::STRING,'N/A'),
        ' to ', COALESCE(ROUND(c.FORECAST_DAY30,0)::STRING,'N/A'),
        '/day over 30 days (peak: ', COALESCE(ROUND(c.FORECAST_PEAK_VALUE,0)::STRING,'N/A'),
        ' on ', COALESCE(c.FORECAST_PEAK_DATE::STRING,'N/A'), ')\n',
        '• Forecast confidence: ', COALESCE(ROUND(c.CI_LOWER_MIN,0)::STRING,'?'),
        ' to ', COALESCE(ROUND(c.CI_UPPER_MAX,0)::STRING,'?'),
        ' (avg band width: ', COALESCE(ROUND(c.CI_AVG_WIDTH,0)::STRING,'?'), ')\n',
        '• Model accuracy (MAPE): ', COALESCE(ROUND(c.MODEL_MAPE,1)::STRING,'N/A'),
        '% — model confidence = ', COALESCE(c.MODEL_CONFIDENCE, 'unknown'), '\n',
        '• Current cases are ', COALESCE(c.PCT_OF_ALL_TIME_PEAK::STRING,'N/A'),
        '% of all-time peak (', COALESCE(ROUND(c.ALL_TIME_PEAK_CASES,0)::STRING,'N/A'),
        '/day on ', COALESCE(c.ALL_TIME_PEAK_DATE::STRING,'N/A'), ')\n',
        '• Anomalies detected: ', c.TOTAL_ANOMALIES::STRING,
        ' (', c.ANOMALY_SPIKES::STRING, ' spikes, ', c.ANOMALY_DROPS::STRING, ' drops)\n\n',
        'For EACH metric, write a short paragraph using a simple real-world analogy.\n',
        'Start each paragraph with the metric name in bold.\n',
        'Include what the ML model predicts and how confident it is.\n',
        'Always reference the actual value. Example: "An Rt of 0.87 means...".\n',
        'No jargon. Separate paragraphs with blank lines.'
    )) AS METRIC_EXPLAINER,

    -- ═══ COLUMN 2: SITUATION SUMMARY ═══
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large', CONCAT(
        'Write a situation briefing for ', c.COUNTRY_REGION,
        '''s health minister. Base EVERY statement on the data below.\n\n',
        'CURRENT DATA (cite these specific numbers in your analysis):\n',
        '• Date: ', c.DATA_DATE::STRING, '\n',
        '• Daily new cases (7-day avg): ', ROUND(c.ROLLING_AVG_7D_CASES,0)::STRING, '\n',
        '• Daily new deaths (7-day avg): ', ROUND(c.ROLLING_AVG_7D_DEATHS,0)::STRING, '\n',
        '• Total confirmed cases: ', c.CUMULATIVE_CONFIRMED::STRING, '\n',
        '• Total deaths: ', c.CUMULATIVE_DEATHS::STRING, '\n',
        '• Rt: ', COALESCE(ROUND(c.RT_EFFECTIVE,2)::STRING, 'N/A'), '\n',
        '• Doubling time: ', COALESCE(ROUND(c.DOUBLING_TIME_DAYS,1)::STRING, 'N/A'), ' days\n',
        '• CFR: ', ROUND(c.CASE_FATALITY_RATE,2)::STRING, '%\n',
        '• Phase: ', c.EPIDEMIC_PHASE, '\n',
        '• Risk: ', c.RISK_TIER, ' (', c.RISK_SCORE::STRING, '/100)\n',
        '• Risk breakdown: Rt=', c.F1_RT::STRING, '/25, Forecast=', c.F2_FORECAST::STRING,
        '/20, Accel=', c.F3_ACCEL::STRING, '/15, Doubling=', c.F4_DOUBLING::STRING,
        '/15, WoW=', c.F5_WOW::STRING, '/10, CFR=', c.F6_CFR::STRING,
        '/10, Volume=', c.F7_VOLUME::STRING, '/5, Anomaly=', c.F8_ANOMALY::STRING, '/5\n',
        '• Acceleration: ', ROUND(c.ACCELERATION,1)::STRING, '\n',
        '\nML FORECAST (Snowflake ML):\n',
        '• Day 1 → Day 30: ', COALESCE(ROUND(c.FORECAST_DAY1,0)::STRING,'N/A'),
        ' → ', COALESCE(ROUND(c.FORECAST_DAY30,0)::STRING,'N/A'), '/day\n',
        '• Forecast peak: ', COALESCE(ROUND(c.FORECAST_PEAK_VALUE,0)::STRING,'N/A'),
        ' on ', COALESCE(c.FORECAST_PEAK_DATE::STRING,'N/A'), '\n',
        '• 95% confidence interval: ', COALESCE(ROUND(c.CI_LOWER_MIN,0)::STRING,'?'),
        ' to ', COALESCE(ROUND(c.CI_UPPER_MAX,0)::STRING,'?'), '\n',
        '• Model accuracy (MAPE): ', COALESCE(ROUND(c.MODEL_MAPE,1)::STRING,'N/A'),
        '% — confidence level: ', COALESCE(c.MODEL_CONFIDENCE,'unknown'), '\n',
        '• Forecast trend: ', ROUND(c.FORECAST_TREND_PCT,1)::STRING, '%\n',
        '\nHISTORICAL CONTEXT:\n',
        '• All-time peak: ', COALESCE(ROUND(c.ALL_TIME_PEAK_CASES,0)::STRING,'N/A'),
        '/day on ', COALESCE(c.ALL_TIME_PEAK_DATE::STRING,'N/A'), '\n',
        '• Current level vs peak: ', COALESCE(c.PCT_OF_ALL_TIME_PEAK::STRING,'N/A'), '%\n',
        '\nANOMALY DETECTION:\n',
        '• Total anomalies: ', c.TOTAL_ANOMALIES::STRING,
        ' (', c.ANOMALY_SPIKES::STRING, ' spikes, ', c.ANOMALY_DROPS::STRING, ' drops)\n',
        '• Largest deviation: ', COALESCE(ROUND(c.MAX_DEVIATION_PCT,0)::STRING,'0'), '%\n',
        CASE WHEN c.LAST_ANOMALY_DATE IS NOT NULL
            THEN CONCAT('• Most recent anomaly: ', c.LAST_ANOMALY_DATE::STRING, '\n')
            ELSE '' END,
        '\nWrite exactly 4 paragraphs:\n',
        'Para 1: Current situation — what is happening NOW. Reference the specific numbers.\n',
        'Para 2: 30-day outlook based on the ML forecast. Mention the confidence range.\n',
        'Para 3: Risk assessment — explain why this country is ', c.RISK_TIER,
        ' and what factors drive the score.\n',
        'Para 4: Two specific recommended actions based on the data.\n\n',
        'RULES: Every claim must reference a specific data point above.\n',
        'Write for a newspaper reader. No jargon. No bullet points.'
    )) AS SITUATION_SUMMARY,

    -- ═══ COLUMN 3: PREVENTIVE MEASURES ═══
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large', CONCAT(
        'Based on ', c.COUNTRY_REGION, '''s current COVID-19 data:\n',
        '• Risk: ', c.RISK_TIER, ' (', c.RISK_SCORE::STRING, '/100)\n',
        '• Rt: ', COALESCE(ROUND(c.RT_EFFECTIVE,2)::STRING, 'N/A'), '\n',
        '• Phase: ', c.EPIDEMIC_PHASE, '\n',
        '• Trend: ', c.TREND_DIRECTION, '\n',
        '• Doubling time: ', COALESCE(ROUND(c.DOUBLING_TIME_DAYS,0)::STRING, 'N/A'), ' days\n',
        '• Forecast trend: ', ROUND(c.FORECAST_TREND_PCT,1)::STRING, '%\n',
        '• Model confidence: ', COALESCE(c.MODEL_CONFIDENCE,'unknown'), '\n',
        '• Current vs all-time peak: ', COALESCE(c.PCT_OF_ALL_TIME_PEAK::STRING,'N/A'), '%\n\n',
        'Recommend 5 preventive measures APPROPRIATE for this risk level.\n',
        'For each measure:\n',
        '1. State the recommendation clearly\n',
        '2. Explain WHY this data supports it (reference the specific numbers above)\n\n',
        'If risk is LOW (score < 25): focus on surveillance and maintenance.\n',
        'If risk is MODERATE (25-49): focus on targeted interventions.\n',
        'If risk is HIGH (≥ 50): focus on urgent escalation measures.\n\n',
        'Ground every recommendation in the actual data. No generic advice.'
    )) AS PREVENTIVE_MEASURES,

    CURRENT_TIMESTAMP() AS GENERATED_AT
FROM PUBLIC_HEALTH_DB.ANALYTICS.VW_CORTEX_CONTEXT c;

-- Add executive brief via SUMMARIZE
ALTER TABLE PUBLIC_HEALTH_DB.ANALYTICS.CORTEX_NARRATIVES
    ADD COLUMN IF NOT EXISTS EXECUTIVE_BRIEF VARCHAR;

UPDATE PUBLIC_HEALTH_DB.ANALYTICS.CORTEX_NARRATIVES
SET EXECUTIVE_BRIEF = SNOWFLAKE.CORTEX.SUMMARIZE(SITUATION_SUMMARY);

-- Verify enhanced narratives
SELECT COUNTRY_REGION, RISK_TIER,
       LEN(METRIC_EXPLAINER) AS ME_LEN,
       LEN(SITUATION_SUMMARY) AS SS_LEN,
       LEN(PREVENTIVE_MEASURES) AS PM_LEN,
       LEN(EXECUTIVE_BRIEF) AS EB_LEN
FROM PUBLIC_HEALTH_DB.ANALYTICS.CORTEX_NARRATIVES
ORDER BY COUNTRY_REGION;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  4.3  CORTEX AI — ANOMALY EXPLANATIONS (TOP 3 PER COUNTRY)    ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ANALYTICS.ANOMALY_EXPLANATIONS AS
WITH top_anomalies AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY COUNTRY_REGION ORDER BY ABS(DEVIATION_PCT) DESC) AS RNK
    FROM PUBLIC_HEALTH_DB.ML.ANOMALIES
    WHERE IS_ANOMALY = TRUE
    QUALIFY RNK <= 3
)
SELECT
    a.COUNTRY_REGION,
    a.DATE,
    a.DAILY_NEW_CASES AS ACTUAL,
    a.EXPECTED,
    a.DEVIATION_PCT,
    a.ANOMALY_DIRECTION,
    DAYNAME(a.DATE) AS DAY_OF_WEEK,
    SNOWFLAKE.CORTEX.COMPLETE(
        'mistral-large',
        CONCAT(
            'You are an epidemiological data analyst. A COVID-19 anomaly was detected for ',
            a.COUNTRY_REGION, ' on ', a.DATE::STRING, ': ',
            'Actual daily cases: ', ROUND(a.DAILY_NEW_CASES, 0)::STRING, '. ',
            'Expected baseline: ', ROUND(a.EXPECTED, 0)::STRING, '. ',
            'Deviation: ', ROUND(a.DEVIATION_PCT, 1)::STRING, '%. ',
            'Direction: ', a.ANOMALY_DIRECTION, '. ',
            'Day of week: ', DAYNAME(a.DATE), '. ',
            'Write exactly 2 sentences for a non-technical official. ',
            'Sentence 1: describe the anomaly in plain English with the numbers. ',
            'Sentence 2: the most likely epidemiological or reporting explanation.'
        )
    ) AS EXPLANATION
FROM top_anomalies a;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  4.4  CORTEX AI — GLOBAL TRIAGE SUMMARY                       ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ANALYTICS.GLOBAL_TRIAGE AS
SELECT
    SNOWFLAKE.CORTEX.COMPLETE(
        'mistral-large',
        CONCAT(
            'You are a public health advisor. Here is the COVID-19 risk assessment for all tracked countries: ',
            (SELECT LISTAGG(
                CONCAT(COUNTRY_REGION, ': ', RISK_TIER, ' (Score ', RISK_SCORE::STRING,
                       ', Rt=', COALESCE(ROUND(RT_EFFECTIVE, 2)::STRING, 'N/A'),
                       ', 7d avg=', ROUND(ROLLING_AVG_7D_CASES, 0)::STRING, ')'),
                '; '
            ) FROM PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS),
            '. Write a 4-sentence global situation summary for a government leader. ',
            'Sentence 1: Overall global trend — are cases rising or falling? ',
            'Sentence 2: Which countries need urgent attention and why (cite their Rt and scores). ',
            'Sentence 3: Which countries are stable and why. ',
            'Sentence 4: One priority action. ',
            'Do not use bullet points. Write in plain English.'
        )
    ) AS GLOBAL_BRIEF,
    CURRENT_TIMESTAMP() AS GENERATED_AT;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  4.6  CORTEX AI — POLICY SIMULATIONS                          ║
-- ╚════════════════════════════════════════════════════════════════╝
-- Three "what-if" scenarios per country, grounded in ML model data

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ANALYTICS.POLICY_SIMULATIONS AS
SELECT
    c.COUNTRY_REGION, c.RISK_TIER,
    SNOWFLAKE.CORTEX.COMPLETE('mistral-large', CONCAT(
        'You are advising ', c.COUNTRY_REGION, '''s government on COVID-19 scenarios.\n\n',
        'CURRENT DATA:\n',
        '• 7-day avg cases: ', ROUND(c.ROLLING_AVG_7D_CASES,0)::STRING, '/day\n',
        '• Rt: ', COALESCE(ROUND(c.RT_EFFECTIVE,2)::STRING, 'N/A'), '\n',
        '• Doubling time: ', COALESCE(ROUND(c.DOUBLING_TIME_DAYS,1)::STRING, 'N/A'), ' days\n',
        '• Phase: ', c.EPIDEMIC_PHASE, '\n',
        '• Risk: ', c.RISK_TIER, ' (', c.RISK_SCORE::STRING, '/100)\n',
        '• Top risk factors: Rt=', c.F1_RT::STRING, '/25, Forecast=', c.F2_FORECAST::STRING,
        '/20, Accel=', c.F3_ACCEL::STRING, '/15, Doubling=', c.F4_DOUBLING::STRING, '/15\n\n',
        'ML MODEL PREDICTIONS:\n',
        '• Forecast Day 1→30: ', COALESCE(ROUND(c.FORECAST_DAY1,0)::STRING,'N/A'),
        ' → ', COALESCE(ROUND(c.FORECAST_DAY30,0)::STRING,'N/A'), '/day\n',
        '• Forecast peak: ', COALESCE(ROUND(c.FORECAST_PEAK_VALUE,0)::STRING,'N/A'),
        ' on ', COALESCE(c.FORECAST_PEAK_DATE::STRING,'N/A'), '\n',
        '• 95% CI: ', COALESCE(ROUND(c.CI_LOWER_MIN,0)::STRING,'?'),
        ' to ', COALESCE(ROUND(c.CI_UPPER_MAX,0)::STRING,'?'), '\n',
        '• Model accuracy (MAPE): ', COALESCE(ROUND(c.MODEL_MAPE,1)::STRING,'N/A'),
        '% — confidence: ', COALESCE(c.MODEL_CONFIDENCE,'unknown'), '\n',
        '• Trend over forecast window: ', ROUND(c.FORECAST_TREND_PCT,1)::STRING, '%\n\n',
        'HISTORICAL CONTEXT:\n',
        '• All-time peak: ', COALESCE(ROUND(c.ALL_TIME_PEAK_CASES,0)::STRING,'N/A'),
        '/day (', COALESCE(c.ALL_TIME_PEAK_DATE::STRING,'N/A'), ')\n',
        '• Current level is ', COALESCE(c.PCT_OF_ALL_TIME_PEAK::STRING,'N/A'),
        '% of that peak\n',
        '• Anomalies detected: ', c.TOTAL_ANOMALIES::STRING,
        ' (', c.ANOMALY_SPIKES::STRING, ' spikes, ', c.ANOMALY_DROPS::STRING, ' drops)\n\n',
        'Using ALL of this data, simulate 3 scenarios for the next 60 days:\n\n',
        'SCENARIO A (Status Quo): Policies stay the same. The ML model predicts ',
        COALESCE(ROUND(c.FORECAST_DAY30,0)::STRING,'N/A'),
        '/day at day 30. What happens by day 60 if this trend continues?\n\n',
        'SCENARIO B (New Variant): Rt increases by 40% to ',
        COALESCE(ROUND(c.RT_EFFECTIVE * 1.4, 2)::STRING, 'N/A'),
        '. How does this change the doubling time and forecast? ',
        'Compare to the all-time peak of ',
        COALESCE(ROUND(c.ALL_TIME_PEAK_CASES,0)::STRING,'N/A'), '/day.\n\n',
        'SCENARIO C (Intervene Now): Interventions reduce Rt by 25% to ',
        COALESCE(ROUND(c.RT_EFFECTIVE * 0.75, 2)::STRING, 'N/A'),
        '. What happens to the forecast trajectory?\n\n',
        'For each scenario: expected case trajectory (cite specific numbers), ',
        'hospital stress level (LOW/MEDIUM/HIGH/CRITICAL), and one specific action.\n',
        'State the model confidence level (', COALESCE(c.MODEL_CONFIDENCE,'unknown'),
        ') when citing predictions.\n',
        'Under 400 words. Plain English.'
    )) AS SCENARIO_ANALYSIS,
    CURRENT_TIMESTAMP() AS GENERATED_AT
FROM PUBLIC_HEALTH_DB.ANALYTICS.VW_CORTEX_CONTEXT c;


SELECT '✅ Phase 4 complete — context view, enhanced narratives, policy simulations generated' AS STATUS;
