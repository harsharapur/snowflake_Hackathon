# Public Pulse — Public Health Intelligence Platform

> Epidemiological intelligence for pandemic decision-making, powered entirely by Snowflake ML and Cortex.

## Demo
**Demo Link:** [Insert Link Here]()

## What It Does

Public Pulse transforms raw COVID-19 case, death, and vaccination data into actionable, data-grounded intelligence for everyday people:
- **Dynamic Text-to-SQL AI Chatbot:** Ask any COVID-19 question in plain English. The AI automatically queries the database using `mistral-large` and explains the ML outputs using analogies and accessible language.
- **30-day Forecasts:** Machine learning predictions with statistically derived 95% confidence intervals.
- **Anomaly Detection:** Identifies statistical outliers (spikes/drops in cases) and flags them for review.
- **Risk Classification:** Comprehensive 8-factor composite scoring (Rt, acceleration, growth trajectories, etc.).
- **Explainable AI:** Provides continuous, dynamic context about what the user is currently viewing to ensure accurate explanations.

## Architecture

Public Pulse uses a 100% Snowflake-native architecture without needing any external processing.

```mermaid
flowchart TD
    subgraph Snowflake Marketplace
        JHU(JHU_COVID_19)
        Vax(OWID_VACCINATIONS)
        Mob(GOOG_MOBILITY)
    end

    subgraph Data Pipeline
        Ingest[01 - Data Ingestion & Joins]
        FeatEng[02 - Feature Engineering\nRt, CFR, Doubling Time]
        ML[03 - Snowflake ML Models\nFORECAST & ANOMALY_DETECTION]
        Risk[04 - Risk Scoring\n8-Factor Composite]
    end

    subgraph User Interface (Streamlit in Snowflake)
        Dash[Interactive Dashboards\nEpidemiology, Forecast, Anomalies]
        Chat[Dynamic AI Chatbot\n2-Step Text-to-SQL]
    end

    JHU --> Ingest
    Vax --> Ingest
    Mob --> Ingest
    Ingest --> FeatEng
    FeatEng --> ML
    ML --> Risk
    Risk --> Dash
    Dash <-->|Query Gen + Explanations| Chat
```

**Chatbot Architecture:** The Chat tab uses a 2-step Cortex pipeline. When a user asks a question, the agent (1) writes a safe `SELECT` statement against the epidemiological/ML tables, executes it, and (2) re-reads the output to explain the raw numbers in plain English.

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
   | 4 | `04_risk_and_cortex.sql` | ~2 min | Evaluates the 8-factor risk scores |

3. **Deploy Dashboard and AI Chatbot**
   - Snowsight → Projects → Streamlit → + Streamlit App
   - Name: `Public Pulse`
   - Database: `PUBLIC_HEALTH_DB`
   - Schema: `ANALYTICS`
   - Warehouse: `SYSTEM$STREAMLIT_NOTEBOOK_WH`
   - Paste contents of `05_streamlit_app.py` → Run
   - Add **`plotly`** via the Packages button (top right)

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
| Cortex only (no external LLMs) | ✅ 2-step Text-to-SQL logic using `CORTEX.COMPLETE('mistral-large')` |
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
