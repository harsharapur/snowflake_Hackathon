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
-- ║  4.2  CORTEX AI — COUNTRY HEALTH BRIEFS                       ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ANALYTICS.CORTEX_NARRATIVES AS
SELECT
    r.COUNTRY_REGION,
    r.RISK_TIER,
    r.RISK_SCORE,
    r.RT_EFFECTIVE,
    r.ROLLING_AVG_7D_CASES,
    r.EPIDEMIC_PHASE,
    r.FORECAST_TREND_PCT,
    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-70b',
        CONCAT(
            'You are a public health analyst. Based on this COVID-19 data for ', r.COUNTRY_REGION, ': ',
            '7-day avg cases: ', ROUND(r.ROLLING_AVG_7D_CASES, 0)::STRING, '. ',
            'Case Fatality Rate: ', ROUND(r.CASE_FATALITY_RATE, 2)::STRING, '%. ',
            'Rt (reproduction number): ', COALESCE(ROUND(r.RT_EFFECTIVE, 2)::STRING, 'unavailable'), '. ',
            'Doubling time: ', COALESCE(ROUND(r.DOUBLING_TIME_DAYS, 1)::STRING, 'N/A'), ' days. ',
            'Epidemic phase: ', r.EPIDEMIC_PHASE, '. ',
            'Risk tier: ', r.RISK_TIER, ' (', r.RISK_SCORE::STRING, '/100). ',
            '30-day forecast trend: ', ROUND(r.FORECAST_TREND_PCT, 1)::STRING, '%. ',
            'Cumulative cases: ', r.CUMULATIVE_CONFIRMED::STRING, '. ',
            'Cumulative deaths: ', r.CUMULATIVE_DEATHS::STRING, '. ',
            'Write exactly 3 sentences for a non-technical government minister. ',
            'Sentence 1: Current situation (mention Rt and phase). ',
            'Sentence 2: 30-day trajectory from the ML forecast. ',
            'Sentence 3: One specific recommended action. ',
            'Do not use technical jargon. Do not use bullet points.'
        )
    ) AS HEALTH_BRIEF,
    CURRENT_TIMESTAMP() AS GENERATED_AT
FROM PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS r;

-- Verify
SELECT COUNTRY_REGION, RISK_TIER, LEFT(HEALTH_BRIEF, 200) AS BRIEF_PREVIEW
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
        'llama3.1-70b',
        CONCAT(
            'You are an epidemiological data analyst. A COVID-19 anomaly was detected for ',
            a.COUNTRY_REGION, ' on ', a.DATE::STRING, ': ',
            'Actual daily cases: ', ROUND(a.DAILY_NEW_CASES, 0)::STRING, '. ',
            'Expected baseline: ', ROUND(a.EXPECTED, 0)::STRING, '. ',
            'Deviation: ', ROUND(a.DEVIATION_PCT, 1)::STRING, '%. ',
            'Direction: ', a.ANOMALY_DIRECTION, '. ',
            'Day of week: ', DAYNAME(a.DATE), '. ',
            'Write exactly 2 sentences. Sentence 1: describe the anomaly statistically. ',
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
        'llama3.1-70b',
        CONCAT(
            'You are a WHO public health advisor. Here is the current COVID-19 risk assessment: ',
            (SELECT LISTAGG(
                CONCAT(COUNTRY_REGION, ': ', RISK_TIER, ' (Score ', RISK_SCORE::STRING,
                       ', Rt=', COALESCE(ROUND(RT_EFFECTIVE, 2)::STRING, 'N/A'),
                       ', 7d avg=', ROUND(ROLLING_AVG_7D_CASES, 0)::STRING, ')'),
                '; '
            ) FROM PUBLIC_HEALTH_DB.ANALYTICS.RISK_TIERS),
            '. Write a 4-sentence global situation summary for the UN Secretary-General. ',
            'Sentence 1: Overall global trend. ',
            'Sentence 2: Which countries need urgent attention and why. ',
            'Sentence 3: Which regions are stable. ',
            'Sentence 4: One priority action for international coordination. ',
            'Do not use bullet points.'
        )
    ) AS GLOBAL_BRIEF,
    CURRENT_TIMESTAMP() AS GENERATED_AT;


SELECT '✅ Phase 4 complete — risk tiers classified, Cortex narratives generated' AS STATUS;
