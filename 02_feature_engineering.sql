-- ============================================================================
-- PUBLIC PULSE — PUBLIC HEALTH INTELLIGENCE PLATFORM
-- FILE 2 OF 4: FEATURE ENGINEERING (25+ FEATURES)
-- ============================================================================
-- Run after 01_setup_and_ingestion.sql
-- Time: ~3 minutes
-- FIX: ROWS BETWEEN clauses are now inline in each function call.
--      Snowflake named WINDOW definitions do not support frame clauses.
-- ============================================================================

USE ROLE TRAINING_ROLE;
USE WAREHOUSE SYSTEM$STREAMLIT_NOTEBOOK_WH;
USE DATABASE PUBLIC_HEALTH_DB;
USE SCHEMA FEATURES;

-- ╔════════════════════════════════════════════════════════════════╗
-- ║  2.1  FULL FEATURE STORE                                      ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES AS
WITH base AS (
    SELECT
        d.*,

        -- ── Rolling Averages (inline ROWS BETWEEN) ──────────────
        ROUND(AVG(DAILY_NEW_CASES)  OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 6  PRECEDING AND CURRENT ROW), 2) AS ROLLING_AVG_7D_CASES,
        ROUND(AVG(DAILY_NEW_CASES)  OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 2) AS ROLLING_AVG_14D_CASES,
        ROUND(AVG(DAILY_NEW_CASES)  OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 2) AS ROLLING_AVG_28D_CASES,
        ROUND(AVG(DAILY_NEW_DEATHS) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 6  PRECEDING AND CURRENT ROW), 2) AS ROLLING_AVG_7D_DEATHS,
        ROUND(AVG(DAILY_NEW_DEATHS) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 2) AS ROLLING_AVG_14D_DEATHS,

        -- ── Rolling Sums ────────────────────────────────────────
        SUM(DAILY_NEW_CASES) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 6  PRECEDING AND CURRENT ROW)            AS SUM_7D_CASES,
        SUM(DAILY_NEW_CASES) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING)            AS PREV_SUM_7D_CASES,
        SUM(DAILY_NEW_CASES) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 27 PRECEDING AND CURRENT ROW)            AS SUM_30D_CASES,

        -- ── Lag values ──────────────────────────────────────────
        LAG(DAILY_NEW_CASES,     7) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE) AS CASES_7D_AGO,
        LAG(CUMULATIVE_CONFIRMED,7) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE) AS CONFIRMED_7D_AGO,

        -- ── Volatility ──────────────────────────────────────────
        ROUND(STDDEV(DAILY_NEW_CASES) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 2) AS VOLATILITY_14D,

        -- ── Case Fatality Rate ──────────────────────────────────
        ROUND(
            CASE WHEN CUMULATIVE_CONFIRMED > 0
                 THEN CUMULATIVE_DEATHS * 100.0 / CUMULATIVE_CONFIRMED
                 ELSE 0 END, 4
        ) AS CASE_FATALITY_RATE,

        -- ── Context ─────────────────────────────────────────────
        DAYOFWEEK(DATE) AS DAY_OF_WEEK,
        DATEDIFF('day',
            FIRST_VALUE(DATE) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE),
            DATE
        ) AS DAYS_SINCE_FIRST_CASE,

        -- ── Rt windows (serial interval ~ 5 days) ───────────────
        SUM(DAILY_NEW_CASES) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS RT_NUMERATOR,
        SUM(DAILY_NEW_CASES) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE ROWS BETWEEN 9 PRECEDING AND 5 PRECEDING)  AS RT_DENOMINATOR

    FROM PUBLIC_HEALTH_DB.RAW.COVID_DAILY d
),
derived AS (
    SELECT
        b.*,

        -- ── Rt ──────────────────────────────────────────────────
        ROUND(
            CASE WHEN RT_DENOMINATOR > 10
                 THEN RT_NUMERATOR::FLOAT / RT_DENOMINATOR
                 ELSE NULL END, 3
        ) AS RT_EFFECTIVE,

        -- ── Doubling Time ────────────────────────────────────────
        ROUND(
            CASE WHEN CONFIRMED_7D_AGO > 0
                  AND CUMULATIVE_CONFIRMED > CONFIRMED_7D_AGO
                 THEN 7.0 * LN(2) / LN(CUMULATIVE_CONFIRMED::FLOAT / CONFIRMED_7D_AGO)
                 ELSE NULL END, 1
        ) AS DOUBLING_TIME_DAYS,

        -- ── Week-over-Week Change (%) ────────────────────────────
        ROUND(
            CASE WHEN PREV_SUM_7D_CASES > 0
                 THEN (SUM_7D_CASES - PREV_SUM_7D_CASES) * 100.0 / PREV_SUM_7D_CASES
                 ELSE 0 END, 2
        ) AS WOW_CHANGE_PCT,

        -- ── Weekly Growth Rate (%) ───────────────────────────────
        ROUND(
            CASE WHEN CASES_7D_AGO > 0
                 THEN (DAILY_NEW_CASES - CASES_7D_AGO) * 100.0 / CASES_7D_AGO
                 ELSE 0 END, 2
        ) AS WEEKLY_GROWTH_RATE,

        -- ── Signal-to-Noise Ratio ────────────────────────────────
        ROUND(
            CASE WHEN VOLATILITY_14D > 0
                 THEN ROLLING_AVG_14D_CASES / VOLATILITY_14D
                 ELSE NULL END, 2
        ) AS SIGNAL_TO_NOISE

    FROM base b
),
classified AS (
    SELECT
        d.*,

        -- ── Acceleration (2nd derivative) ───────────────────────
        ROUND(
            ROLLING_AVG_7D_CASES - LAG(ROLLING_AVG_7D_CASES, 7) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE),
            2
        ) AS ACCELERATION,

        -- ── Trend Direction ──────────────────────────────────────
        CASE
            WHEN ROLLING_AVG_7D_CASES > ROLLING_AVG_14D_CASES * 1.1 THEN 'ACCELERATING'
            WHEN ROLLING_AVG_7D_CASES < ROLLING_AVG_14D_CASES * 0.9 THEN 'DECELERATING'
            ELSE 'STABLE'
        END AS TREND_DIRECTION,

        -- ── Epidemic Phase (6-class) ─────────────────────────────
        CASE
            WHEN ROLLING_AVG_7D_CASES > ROLLING_AVG_14D_CASES * 1.2
             AND ROLLING_AVG_7D_CASES > ROLLING_AVG_28D_CASES * 1.3  THEN 'EXPONENTIAL_GROWTH'
            WHEN ROLLING_AVG_7D_CASES > ROLLING_AVG_14D_CASES * 1.05 THEN 'LINEAR_GROWTH'
            WHEN ABS(ROLLING_AVG_7D_CASES - ROLLING_AVG_14D_CASES)
                 <= ROLLING_AVG_14D_CASES * 0.05                      THEN 'PLATEAU'
            WHEN ROLLING_AVG_7D_CASES < ROLLING_AVG_14D_CASES * 0.95
             AND ROLLING_AVG_7D_CASES > ROLLING_AVG_28D_CASES * 0.7  THEN 'LINEAR_DECLINE'
            WHEN ROLLING_AVG_7D_CASES < ROLLING_AVG_28D_CASES * 0.7  THEN 'EXPONENTIAL_DECLINE'
            ELSE 'TRANSITION'
        END AS EPIDEMIC_PHASE,

        -- ── Endemic Flag placeholder (updated below) ─────────────
        FALSE AS ENDEMIC_FLAG

    FROM derived d
)
SELECT * FROM classified
ORDER BY COUNTRY_REGION, DATE;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  2.2  ENDEMIC FLAG UPDATE                                     ║
-- ╚════════════════════════════════════════════════════════════════╝
-- Countries where ALL of the last 60 days are in a decline phase.

UPDATE PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
SET ENDEMIC_FLAG = TRUE
WHERE COUNTRY_REGION IN (
    SELECT COUNTRY_REGION
    FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
    WHERE DATE >= (SELECT DATEADD('day', -60, MAX(DATE)) FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES)
    GROUP BY COUNTRY_REGION
    HAVING SUM(CASE WHEN EPIDEMIC_PHASE IN ('EXPONENTIAL_DECLINE','LINEAR_DECLINE') THEN 1 ELSE 0 END)
         = COUNT(*)
);


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  2.3  VERIFICATION                                            ║
-- ╚════════════════════════════════════════════════════════════════╝

SELECT 'COVID_FEATURES' AS TBL, COUNT(*) AS ROWS_,
       COUNT(DISTINCT COUNTRY_REGION) AS COUNTRIES,
       COUNT(DISTINCT EPIDEMIC_PHASE) AS PHASES
FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES;

-- Latest row per country — confirm all key features populated
SELECT
    COUNTRY_REGION,
    DATE,
    ROUND(ROLLING_AVG_7D_CASES, 0)   AS AVG7D,
    ROUND(RT_EFFECTIVE, 2)           AS RT,
    ROUND(DOUBLING_TIME_DAYS, 1)     AS DBL_TIME,
    EPIDEMIC_PHASE,
    ENDEMIC_FLAG
FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES
WHERE DATE = (SELECT MAX(DATE) FROM PUBLIC_HEALTH_DB.FEATURES.COVID_FEATURES)
ORDER BY AVG7D DESC;

SELECT '✅ Phase 2 complete — 25+ features engineered' AS STATUS;
