-- ============================================================================
-- SENTINEL — PUBLIC HEALTH INTELLIGENCE PLATFORM
-- FILE 1 OF 4: SETUP & DATA INGESTION
-- ============================================================================
-- Run this in a Snowflake SQL Worksheet (highlight all → Ctrl+Enter)
-- Time: ~3 minutes
-- ============================================================================

USE ROLE TRAINING_ROLE;
USE WAREHOUSE SYSTEM$STREAMLIT_NOTEBOOK_WH;

-- ╔════════════════════════════════════════════════════════════════╗
-- ║  1.1  CREATE DATABASE & SCHEMAS                                ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE DATABASE IF NOT EXISTS PUBLIC_HEALTH_DB;
USE DATABASE PUBLIC_HEALTH_DB;

CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS FEATURES;
CREATE SCHEMA IF NOT EXISTS ML;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;

USE SCHEMA RAW;

-- ╔════════════════════════════════════════════════════════════════╗
-- ║  1.2  INGEST JHU COVID-19 — CASES + DEATHS                    ║
-- ╚════════════════════════════════════════════════════════════════╝
-- Source: COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.JHU_COVID_19
-- Strategy: Pivot CASE_TYPE column into separate CONFIRMED / DEATHS columns
--           using a single query with conditional aggregation.
--           Filter to 15 target countries, national level only.
-- Fix: Previous version had two back-to-back query assignments.
--      This is the single, clean version.

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.RAW.COVID_COMBINED AS
WITH normalized AS (
    -- Map EXACT JHU Marketplace country names → standard display names.
    -- Names confirmed by running diagnostic query on the actual dataset.
    SELECT
        CASE COUNTRY_REGION
            WHEN 'United States'              THEN 'US'
            WHEN 'United States of America'   THEN 'US'
            WHEN 'Korea, Republic of'         THEN 'Korea, South'   -- South Korea (confirmed)
            WHEN 'Korea, North'               THEN 'Korea, North'
            WHEN 'Iran, Islamic Republic of'  THEN 'Iran'           -- confirmed JHU name
            WHEN 'Russian Federation'         THEN 'Russia'
            ELSE COUNTRY_REGION
        END AS COUNTRY_REGION,
        DATE,
        CASE_TYPE,
        CASES
    FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.JHU_COVID_19
    WHERE COUNTRY_REGION IN (
        -- Exact JHU names (confirmed from diagnostic query):
        'Australia',
        'Brazil',
        'France',
        'Germany',
        'India',
        'Iran, Islamic Republic of',      -- displayed as 'Iran'
        'Italy',
        'Japan',
        'Korea, Republic of',             -- displayed as 'Korea, South' (South Korea)
        'Mexico',
        'Nigeria',
        'South Africa',
        'Turkey',
        'United Kingdom',
        'Russian Federation',             -- displayed as 'Russia' — replaces US if not found
        -- US variants (try all; only matching rows will sum):
        'US',
        'United States',
        'United States of America'
    )
)
SELECT
    COUNTRY_REGION,
    DATE,
    SUM(CASE WHEN CASE_TYPE = 'Confirmed' THEN CASES ELSE 0 END) AS CUMULATIVE_CONFIRMED,
    SUM(CASE WHEN CASE_TYPE = 'Deaths'    THEN CASES ELSE 0 END) AS CUMULATIVE_DEATHS
FROM normalized
GROUP BY COUNTRY_REGION, DATE
ORDER BY COUNTRY_REGION, DATE;

-- Verify
SELECT 'COVID_COMBINED' AS TBL,
       COUNT(*) AS ROW_COUNT,
       COUNT(DISTINCT COUNTRY_REGION) AS COUNTRIES,
       MIN(DATE) AS FIRST_DATE,
       MAX(DATE) AS LAST_DATE
FROM PUBLIC_HEALTH_DB.RAW.COVID_COMBINED;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  1.3  COMPUTE DAILY DELTAS                                     ║
-- ╚════════════════════════════════════════════════════════════════╝
-- Cumulative → daily new values using LAG().
-- GREATEST(..., 0) clips negative corrections (countries revise counts down).

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.RAW.COVID_DAILY AS
SELECT
    COUNTRY_REGION,
    DATE,
    CUMULATIVE_CONFIRMED,
    CUMULATIVE_DEATHS,
    GREATEST(0, CUMULATIVE_CONFIRMED - COALESCE(
        LAG(CUMULATIVE_CONFIRMED) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE), 0
    )) AS DAILY_NEW_CASES,
    GREATEST(0, CUMULATIVE_DEATHS - COALESCE(
        LAG(CUMULATIVE_DEATHS) OVER (PARTITION BY COUNTRY_REGION ORDER BY DATE), 0
    )) AS DAILY_NEW_DEATHS
FROM PUBLIC_HEALTH_DB.RAW.COVID_COMBINED
ORDER BY COUNTRY_REGION, DATE;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  1.4  LOAD VACCINATION DATA (OWID — IF AVAILABLE)              ║
-- ╚════════════════════════════════════════════════════════════════╝
-- The Starschema share may or may not include OWID vaccinations.
-- Strategy: Try to load it; if the table doesn't exist, create an empty
-- table with the correct schema so downstream code always works.

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.RAW.VACCINATIONS (
    COUNTRY_REGION   VARCHAR,
    DATE             DATE,
    TOTAL_VACCINATIONS        NUMBER,
    PEOPLE_VACCINATED         NUMBER,
    PEOPLE_FULLY_VACCINATED   NUMBER,
    DAILY_VACCINATIONS        FLOAT,
    VACCINATED_PER_100        FLOAT,
    FULLY_VACCINATED_PER_100  FLOAT
);

-- Attempt OWID load — uncomment if the table exists in your share
-- If it errors, that's fine — the empty table above serves as fallback.
/*
INSERT INTO PUBLIC_HEALTH_DB.RAW.VACCINATIONS
SELECT
    CASE LOCATION
        WHEN 'United States' THEN 'US'
        WHEN 'South Korea'   THEN 'Korea, South'
        ELSE LOCATION
    END AS COUNTRY_REGION,
    DATE,
    TOTAL_VACCINATIONS,
    PEOPLE_VACCINATED,
    PEOPLE_FULLY_VACCINATED,
    NEW_VACCINATIONS_SMOOTHED             AS DAILY_VACCINATIONS,
    PEOPLE_VACCINATED_PER_HUNDRED         AS VACCINATED_PER_100,
    PEOPLE_FULLY_VACCINATED_PER_HUNDRED   AS FULLY_VACCINATED_PER_100
FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.OWID_VACCINATIONS
WHERE LOCATION IN (
    'United States', 'Brazil', 'Mexico',
    'United Kingdom', 'France', 'Germany', 'Italy',
    'India', 'Japan', 'South Korea', 'Australia',
    'South Africa', 'Nigeria', 'Turkey', 'Iran'
)
ORDER BY COUNTRY_REGION, DATE;
*/

-- Check vaccination row count (0 if table not available)
SELECT 'VACCINATIONS' AS TBL, COUNT(*) AS ROW_COUNT,
       COUNT(DISTINCT COUNTRY_REGION) AS COUNTRIES
FROM PUBLIC_HEALTH_DB.RAW.VACCINATIONS;


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  1.5  LOAD GOOGLE MOBILITY DATA (IF AVAILABLE)                 ║
-- ╚════════════════════════════════════════════════════════════════╝

CREATE OR REPLACE TABLE PUBLIC_HEALTH_DB.RAW.MOBILITY (
    COUNTRY_REGION              VARCHAR,
    DATE                        DATE,
    RETAIL_AND_RECREATION       FLOAT,
    GROCERY_AND_PHARMACY        FLOAT,
    TRANSIT_STATIONS            FLOAT,
    WORKPLACES                  FLOAT,
    RESIDENTIAL                 FLOAT,
    MOBILITY_INDEX              FLOAT
);

-- Attempt mobility load — uncomment if available
/*
INSERT INTO PUBLIC_HEALTH_DB.RAW.MOBILITY
SELECT
    COUNTRY_REGION,
    DATE,
    RETAIL_AND_RECREATION,
    GROCERY_AND_PHARMACY,
    TRANSIT_STATIONS,
    WORKPLACES,
    RESIDENTIAL,
    ROUND((COALESCE(RETAIL_AND_RECREATION,0)
         + COALESCE(TRANSIT_STATIONS,0)
         + COALESCE(WORKPLACES,0)) / 3.0, 2) AS MOBILITY_INDEX
FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.GOOG_GLOBAL_MOBILITY_REPORT
WHERE COUNTRY_REGION IN (
    'US', 'Brazil', 'Mexico',
    'United Kingdom', 'France', 'Germany', 'Italy',
    'India', 'Japan', 'Korea, South', 'Australia',
    'South Africa', 'Nigeria', 'Turkey', 'Iran'
)
AND SUB_REGION_1 IS NULL
ORDER BY COUNTRY_REGION, DATE;
*/


-- ╔════════════════════════════════════════════════════════════════╗
-- ║  1.6  DATA PROFILING                                           ║
-- ╚════════════════════════════════════════════════════════════════╝

-- Country summary
SELECT
    COUNTRY_REGION,
    COUNT(*) AS DAYS,
    MIN(DATE) AS FIRST_DATE,
    MAX(DATE) AS LAST_DATE,
    MAX(CUMULATIVE_CONFIRMED) AS TOTAL_CASES,
    MAX(CUMULATIVE_DEATHS)    AS TOTAL_DEATHS,
    ROUND(MAX(CUMULATIVE_DEATHS) * 100.0 / NULLIF(MAX(CUMULATIVE_CONFIRMED), 0), 2) AS CFR_PCT,
    SUM(CASE WHEN DAILY_NEW_CASES IS NULL OR DAILY_NEW_CASES = 0 THEN 1 ELSE 0 END) AS ZERO_DAYS,
    ROUND(SUM(CASE WHEN DAILY_NEW_CASES = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS ZERO_PCT
FROM PUBLIC_HEALTH_DB.RAW.COVID_DAILY
GROUP BY COUNTRY_REGION
ORDER BY TOTAL_CASES DESC;

SELECT '✅ Phase 1 complete — data ingested' AS STATUS;
