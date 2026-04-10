-- ============================================================================
-- PUBLIC PULSE — PUBLIC HEALTH INTELLIGENCE PLATFORM
-- FILE 3 OF 4: ML MODELS (FORECAST + ANOMALY DETECTION)
-- ============================================================================
-- Run after 02_feature_engineering.sql
-- Time: ~15-20 minutes (ML training is compute-intensive)
-- ============================================================================

USE ROLE TRAINING_ROLE;
USE WAREHOUSE SYSTEM$STREAMLIT_NOTEBOOK_WH;
USE DATABASE PUBLIC_HEALTH_DB;
USE SCHEMA ML;

-- ╔════════════════════════════════════════════════════════════════╗
-- ║  3.1  PREPARE TRAINING DATA — FIXED CALENDAR SPLIT            ║
-- ╚════════════════════════════════════════════════════════════════╝
-- Train: Jan 2020 – Mar 31, 2022 (~800 days per country)
-- Every country gets the SAME training/validation cutoff.

CREATE OR REPLACE VIEW PUBLIC_HEALTH_DB.ML.FORECAST_TRAINING AS
SELECT
    COUNTRY_REGION,
    DATE,
    ROLLING_AVG_7D_CASES AS TARGET
FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
WHERE DATE >= '2020-03-15'
  AND DATE <= '2022-03-31'
  AND ROLLING_AVG_7D_CASES IS NOT NULL
ORDER BY COUNTRY_REGION, DATE;

-- Verify
SELECT COUNTRY_REGION, COUNT(*) AS TRAIN_DAYS,
       MIN(DATE) AS START_DT, MAX(DATE) AS END_DT
FROM PUBLIC_HEALTH_DB.ML.FORECAST_TRAINING
GROUP BY COUNTRY_REGION
ORDER BY COUNTRY_REGION;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  3.2  TRAIN SNOWFLAKE.ML.FORECAST                              ║
-- ╚════════════════════════════════════════════════════════════════╝
-- KEY FIX: Use TABLE() wrapper instead of SYSTEM$REFERENCE()
--          SYSTEM$REFERENCE is deprecated/broken on some editions.

CREATE OR REPLACE SNOWFLAKE.ML.FORECAST covid_cases_model(
    INPUT_DATA => TABLE(PUBLIC_HEALTH_DB.ML.FORECAST_TRAINING),
    SERIES_COLNAME => 'COUNTRY_REGION',
    TIMESTAMP_COLNAME => 'DATE',
    TARGET_COLNAME => 'TARGET'
);


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  3.3  GENERATE 30-DAY FORWARD FORECAST                        ║
-- ╚════════════════════════════════════════════════════════════════╝

CALL covid_cases_model!FORECAST(
    FORECASTING_PERIODS => 30,
    CONFIG_OBJECT => {'prediction_interval': 0.95}
);

-- Store results
CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ML.FORECASTS AS
SELECT
    SERIES                AS COUNTRY_REGION,
    TS::DATE              AS DATE,
    ROUND(FORECAST, 2)    AS FORECASTED_CASES,
    ROUND(LOWER_BOUND, 2) AS FORECAST_LOWER,
    ROUND(UPPER_BOUND, 2) AS FORECAST_UPPER,
    'FORECAST'            AS TYPE
FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Verify
SELECT COUNTRY_REGION, COUNT(*) AS DAYS,
       MIN(DATE) AS FC_START, MAX(DATE) AS FC_END,
       ROUND(AVG(FORECASTED_CASES), 0) AS AVG_FC,
       ROUND(AVG(FORECAST_UPPER - FORECAST_LOWER), 0) AS AVG_CI_WIDTH
FROM PUBLIC_HEALTH_DB.ML.FORECASTS
GROUP BY COUNTRY_REGION
ORDER BY COUNTRY_REGION;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  3.4  MODEL EVALUATION METRICS                                 ║
-- ╚════════════════════════════════════════════════════════════════╝

CALL covid_cases_model!SHOW_EVALUATION_METRICS();

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ML.FORECAST_METRICS AS
SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  3.5  HISTORY + FORECAST COMBINED TABLE                        ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ML.HISTORY_FORECAST AS
SELECT
    COUNTRY_REGION,
    DATE,
    ROLLING_AVG_7D_CASES AS VAL,
    NULL::FLOAT          AS LOWER_BOUND,
    NULL::FLOAT          AS UPPER_BOUND,
    'ACTUAL'             AS TYPE
FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
WHERE DATE >= '2020-03-15'

UNION ALL

SELECT
    COUNTRY_REGION,
    DATE,
    FORECASTED_CASES     AS VAL,
    FORECAST_LOWER       AS LOWER_BOUND,
    FORECAST_UPPER       AS UPPER_BOUND,
    'FORECAST'           AS TYPE
FROM PUBLIC_HEALTH_DB.ML.FORECASTS

ORDER BY COUNTRY_REGION, DATE;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  3.6  ANOMALY DETECTION                                        ║
-- ╚════════════════════════════════════════════════════════════════╝
-- TWO OPTIONS: Try A first; if it fails, use B.
-- Both produce the SAME output schema for the Streamlit app.

-- =================================================================
-- OPTION A: SNOWFLAKE.ML.ANOMALY_DETECTION (preferred)
-- Uses TABLE() wrapper. Remove the /* */ comments to enable.
-- =================================================================

/*
CREATE OR REPLACE VIEW PUBLIC_HEALTH_DB.ML.ANOMALY_INPUT AS
SELECT
    COUNTRY_REGION,
    DATE,
    DAILY_NEW_CASES AS TARGET
FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
WHERE DATE >= '2020-03-15'
  AND DAILY_NEW_CASES IS NOT NULL
ORDER BY COUNTRY_REGION, DATE;

CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION covid_anomaly_model(
    INPUT_DATA        => TABLE(PUBLIC_HEALTH_DB.ML.ANOMALY_INPUT),
    SERIES_COLNAME    => 'COUNTRY_REGION',
    TIMESTAMP_COLNAME => 'DATE',
    TARGET_COLNAME    => 'TARGET'
);

CALL covid_anomaly_model!DETECT_ANOMALIES(
    INPUT_DATA        => TABLE(PUBLIC_HEALTH_DB.ML.ANOMALY_INPUT),
    TIMESTAMP_COLNAME => 'DATE',
    TARGET_COLNAME    => 'TARGET',
    SERIES_COLNAME    => 'COUNTRY_REGION'
);

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ML.ANOMALIES AS
SELECT
    SERIES              AS COUNTRY_REGION,
    TS::DATE            AS DATE,
    Y                   AS DAILY_NEW_CASES,
    FORECAST            AS EXPECTED,
    IS_ANOMALY,
    PERCENTILE,
    DISTANCE,
    ROUND(
        CASE WHEN FORECAST IS NOT NULL AND FORECAST > 0
             THEN (Y - FORECAST) * 100.0 / FORECAST
             ELSE 0 END, 1
    ) AS DEVIATION_PCT,
    CASE
        WHEN Y > FORECAST THEN 'SPIKE'
        WHEN Y < FORECAST THEN 'DROP'
        ELSE 'NEUTRAL'
    END AS ANOMALY_DIRECTION
FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
*/


-- =================================================================
-- OPTION B: SQL Z-SCORE ANOMALY DETECTION (always works)
-- Uses 14-day rolling mean/stddev. |Z| > 2.5 = anomaly.
-- This is ACTIVE by default since ML.ANOMALY_DETECTION may not
-- be available on all account editions.
-- =================================================================

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.ML.ANOMALIES AS
WITH rolling_base AS (
    -- Step 1: Compute rolling mean and mean-of-squares (both support ROWS BETWEEN)
    SELECT
        COUNTRY_REGION,
        DATE,
        DAILY_NEW_CASES,
        AVG(DAILY_NEW_CASES) OVER (
            PARTITION BY COUNTRY_REGION ORDER BY DATE
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS ROLLING_MEAN,
        -- E[x²] for variance formula: Var = E[x²] - E[x]²
        AVG(POWER(DAILY_NEW_CASES, 2)) OVER (
            PARTITION BY COUNTRY_REGION ORDER BY DATE
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS ROLLING_MEAN_SQ
    FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
    WHERE DATE >= '2020-03-15'
      AND DAILY_NEW_CASES IS NOT NULL
),
rolling AS (
    -- Step 2: Derive std from variance. σ = √(E[x²] - E[x]²). GREATEST clips float rounding.
    SELECT *,
        SQRT(GREATEST(ROLLING_MEAN_SQ - POWER(ROLLING_MEAN, 2), 0)) AS ROLLING_STD
    FROM rolling_base
),
scored AS (
    SELECT *,
        CASE WHEN ROLLING_STD > 0
             THEN (DAILY_NEW_CASES - ROLLING_MEAN) / ROLLING_STD
             ELSE 0 END AS Z_SCORE
    FROM rolling
)
SELECT
    COUNTRY_REGION,
    DATE,
    DAILY_NEW_CASES,
    ROUND(ROLLING_MEAN, 2)         AS EXPECTED,
    ABS(Z_SCORE) > 2.5             AS IS_ANOMALY,
    NULL::FLOAT                    AS PERCENTILE,
    ROUND(Z_SCORE, 3)              AS DISTANCE,
    ROUND(
        CASE WHEN ROLLING_MEAN > 0
             THEN (DAILY_NEW_CASES - ROLLING_MEAN) * 100.0 / ROLLING_MEAN
             ELSE 0 END, 1
    )                              AS DEVIATION_PCT,
    CASE WHEN DAILY_NEW_CASES > ROLLING_MEAN THEN 'SPIKE'
         WHEN DAILY_NEW_CASES < ROLLING_MEAN THEN 'DROP'
         ELSE 'NEUTRAL'
    END                            AS ANOMALY_DIRECTION
FROM scored
ORDER BY COUNTRY_REGION, DATE;


-- Anomaly summary
SELECT
    COUNTRY_REGION,
    SUM(CASE WHEN IS_ANOMALY THEN 1 ELSE 0 END)                                 AS TOTAL_ANOMALIES,
    SUM(CASE WHEN IS_ANOMALY AND ANOMALY_DIRECTION = 'SPIKE' THEN 1 ELSE 0 END) AS SPIKES,
    SUM(CASE WHEN IS_ANOMALY AND ANOMALY_DIRECTION = 'DROP'  THEN 1 ELSE 0 END) AS DROPS,
    ROUND(MAX(CASE WHEN IS_ANOMALY THEN ABS(DEVIATION_PCT) END), 1)              AS MAX_DEVIATION
FROM PUBLIC_HEALTH_DB.ML.ANOMALIES
GROUP BY COUNTRY_REGION
ORDER BY TOTAL_ANOMALIES DESC;

SELECT '✅ Phase 3 complete — forecast + anomalies generated' AS STATUS;
