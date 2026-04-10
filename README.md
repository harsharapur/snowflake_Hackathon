# Public Pulse — Public Health Intelligence Platform

> Epidemiological intelligence for pandemic decision-making, powered entirely by Snowflake.

## What It Does

Transforms raw COVID-19 case, death, and vaccination data into actionable intelligence:
- **30-day forecasts** with statistically derived confidence intervals
- **Anomaly detection** (spike vs. drop classification)
- **Risk classification** (8-factor composite scoring)
- **AI-generated health briefs** via Snowflake Cortex
- **Live Q&A** — ask Cortex any question about any country

## Architecture

```
┌──────────────────────────────────────────────┐
│  Snowflake Marketplace (Starschema)          │
│  JHU_COVID_19 │ GOOG_MOBILITY │ OWID_VAX    │
└──────────────────┬───────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  01 Ingestion (SQL)         │
    │  15 countries, daily deltas │
    │  Cases + deaths + vax       │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  02 Feature Engineering     │
    │  25+ features: Rt, phase,   │
    │  doubling time, CFR,        │
    │  acceleration, SNR          │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  03 ML Models               │
    │  SNOWFLAKE.ML.FORECAST      │
    │  SNOWFLAKE.ML.ANOMALY_DET   │
    │  Fixed calendar split       │
    │  Native 95% CI              │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  04 Risk + Cortex           │
    │  8-factor risk (0-100)      │
    │  CORTEX.COMPLETE narratives │
    │  Anomaly explanations       │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  05 Streamlit in Snowflake  │
    │  7 interactive tabs         │
    │  Maroon/white design        │
    │  Live Cortex Q&A            │
    └─────────────────────────────┘
```

## Setup Instructions

### Prerequisites
- Snowflake trial account with `TRAINING_ROLE` and `SYSTEM$STREAMLIT_NOTEBOOK_WH`
- Starschema COVID-19 Epidemiological Data from Marketplace (free)

### Step-by-Step

1. **Mount Marketplace Data**
   - Snowsight → Data Products → Marketplace → Search "Starschema COVID-19"
   - Click "Get" → creates `COVID19_EPIDEMIOLOGICAL_DATA` database

2. **Run SQL Pipeline** (4 files, in order)
   - Snowsight → Projects → Worksheets → + SQL Worksheet
   - Paste and run each file (highlight all → Ctrl+Enter):

   | Step | File | Time | What It Does |
   |------|------|------|-------------|
   | 1 | `01_setup_and_ingestion.sql` | ~3 min | Creates DB, loads data from 3 tables |
   | 2 | `02_feature_engineering.sql` | ~3 min | Computes 25+ epidemiological features |
   | 3 | `03_ml_models.sql` | ~15-20 min | Trains FORECAST + ANOMALY_DETECTION |
   | 4 | `04_risk_and_cortex.sql` | ~5 min | Risk scores + Cortex AI narratives |

3. **Deploy Dashboard**
   - Snowsight → Projects → Streamlit → + Streamlit App
   - Name: `Public Pulse`
   - Database: `PUBLIC_HEALTH_DB`
   - Schema: `ANALYTICS`
   - Warehouse: `SYSTEM$STREAMLIT_NOTEBOOK_WH`
   - Paste contents of `05_streamlit_app.py` → Run
   - Add `plotly` via the Packages button (top right)

## Data Sources

| Table | Source | Records | Purpose |
|-------|--------|---------|---------|
| `JHU_COVID_19` | Johns Hopkins CSSE | ~17K rows | Daily cases + deaths |
| `GOOG_GLOBAL_MOBILITY_REPORT` | Google | Optional | Movement patterns |
| `OWID_VACCINATIONS` | Our World in Data | Optional | Vaccination rates |

All from a single Marketplace listing — no external downloads.

## ML Models

| Model | Engine | Split | Metric |
|-------|--------|-------|--------|
| Forecast | `SNOWFLAKE.ML.FORECAST` | Train ≤ 2022-03-31 | MAPE via `SHOW_EVALUATION_METRICS()` |
| Anomaly | `SNOWFLAKE.ML.ANOMALY_DETECTION` | Full series | Statistical threshold |
| Risk | SQL deterministic | Latest snapshot | 8-factor composite (0–100) |

**Confidence intervals** are native from Snowflake ML — not hardcoded.

## Hackathon Compliance

| Requirement | Status |
|------------|--------|
| Snowflake Marketplace data only | ✅ Starschema COVID-19 |
| All ML in Snowflake | ✅ SNOWFLAKE.ML.FORECAST + ANOMALY_DETECTION |
| Cortex only (no external LLMs) | ✅ CORTEX.COMPLETE('llama3.1-70b') |
| Streamlit in Snowflake | ✅ SiS deployment |
| 10+ countries | ✅ 15 countries, 6 WHO regions |
| Fairness documentation | ✅ Methodology tab |
| Open-source libraries | ✅ plotly (MIT license) |

## Countries (15 across 6 WHO Regions)

| Region | Countries |
|--------|-----------|
| Americas | US, Brazil, Mexico |
| Europe | UK, France, Germany, Italy |
| SE Asia | India |
| W Pacific | Japan, South Korea, Australia |
| Africa | South Africa, Nigeria |
| E Mediterranean | Turkey, Iran |

## License

For educational and hackathon purposes. COVID-19 data is CC BY 4.0 (JHU CSSE).
