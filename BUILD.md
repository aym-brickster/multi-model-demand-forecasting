# How This Was Built — Build Process & AI Mindset

> **Author's note:** this is my account of how I built this solution and where AI tooling
> changed the way the work got done. Review and adjust any wording so it matches your own
> voice before submitting — the facts and file references are accurate to the repo.

## What I built

Starting from the open-source [Many-Model Forecasting (MMF) solution accelerator](https://github.com/databricks-industry-solutions/many-model-forecasting),
I built an **end-to-end demand-forecasting solution on Databricks** and extended the library
itself. The solution runs the full journey — raw data → governed Delta tables → backtested
model training → production scoring → low-latency serving → a conversational query surface →
an interactive app:

- **Lakeflow / ingestion & transformation** — a monthly BASF-chemicals demand dataset (12 product
  families, 5 years, with exogenous regressors) landed as a governed Delta table.
- **Unity Catalog governance** — table/column comments and discoverability tags applied and
  verified via `information_schema`.
- **MLflow modeling & backtesting** — **17 models × 99 time series × 10 backtest windows =
  16,830 evaluations** (MLflow run `83517d0d-584f-43f6-b505-a442fccdf763`), best model selected
  per series, models registered in Unity Catalog.
- **Lakebase (PostgreSQL 17) serving** — evaluation, scoring, and training tables synced for
  sub-second reads.
- **Genie Agent** — a conversational natural-language surface over the governed forecast tables.
- **Dash app** (`apps/app.py`) — reads from Lakebase for an interactive leaderboard + forecast view.

## The workflow, stage by stage — and where AI did the work

My primary AI tool throughout was **Databricks Genie / the Databricks Assistant** (including the
MMF Agent skills, which run on Genie Code). My loop was deliberately *iterative and
conversational* — "vibe coding": I described the outcome I wanted in natural language, let the
assistant draft the Spark/SQL/SDK code against the real workspace, ran it, read the output, and
corrected course. Concretely, by stage:

| Stage | What I did | Where AI multiplied the work |
|-------|------------|------------------------------|
| **1. Domain data** | Generated a realistic BASF-chemicals monthly series with seasonality, trend and three exogenous regressors (`examples/generate_basf_chemicals_data.py`, `evidence_runner`). | Assistant drafted the synthetic-data generator and the exogenous-regressor logic from a plain-language spec; I tuned the ranges. |
| **2. Governance** | Applied `COMMENT ON TABLE` / `ALTER COLUMN … COMMENT` and tags, then verified via `information_schema`. | Assistant wrote the governance + verification SQL and handled the tag-policy constraints the workspace rejected on the first pass. |
| **3. Modeling & backtest** | Configured the model set and backtest in `mmf_sa/forecasting_conf_chemicals.yaml`; ran `run_forecast` across 17 models. | Assistant helped wire the config-over-code setup and interpret the backtest leaderboard. |
| **4. Library extensions** | Added **conformal prediction intervals** and **negative-forecast handling** to `mmf_sa` (see trade-offs below). | Assistant drafted the split-conformal implementation; I decided the method and where it plugged into the pipeline. |
| **5. Serving** | Stood up a Lakebase project and synced the output tables (`examples/setup_lakebase.py`). | Assistant generated the SDK calls for endpoint creation, credential handling and synced-table status checks. |
| **6. Genie + App** | Created the Genie space (`examples/setup_genie_agent.py`) and the Dash app (`apps/app.py` + `apps/lakebase_client.py`). | Assistant scaffolded the Genie setup and the Lakebase-backed app queries. |
| **7. Evidence capture** | Built `examples/evidence_runner` to execute every layer live and write a durable run log. | Assistant helped design the capture so outputs survive Git (plain Markdown) rather than being stripped from notebook cells. |

**Where AI genuinely changed how the work got done:** the biggest multiplier was collapsing the
"look up the SDK/SQL API → write boilerplate → debug permissions" cycle. Governance SQL, the
Lakebase `psycopg`/SDK plumbing, the Genie conversation API, and the evidence-capture notebook
were all drafted in natural language against the live workspace, so I spent my time on
*decisions* (which models, how to size intervals, how to handle zeros/negatives, what evidence
to capture) instead of on syntax. The trade-off was that assistant-drafted code needed review —
e.g. the first governance pass used tag keys the workspace policy rejected, and I had to
constrain it to allowed values.

## Key technical decisions & trade-offs (mine)

1. **Conformal prediction intervals** (`mmf_sa/models/abstract_model.py`, `compute_conformal_intervals`).
   Rather than rely on each model's native (and inconsistent) interval, I added a **distribution-free
   split-conformal** method: accumulate absolute backtest residuals, take the `(1-alpha)` quantile as
   the interval half-width. Trade-off: it needs enough backtest windows to be stable, and it widens
   intervals for noisy/intermittent series — acceptable for a demand use case where honest uncertainty
   matters more than tight bands. Gated by `prediction_interval_level` so it stays opt-in.

2. **Negative-forecast handling** (`mmf_sa/Forecaster.py:348`). Demand can't be negative, but several
   statistical models will happily predict below zero on low-volume/intermittent series. I clip
   forecasts at 0 **unless** `allow_negative_values` is set. Trade-off: clipping introduces a small
   positive bias on near-zero series; I judged that preferable to shipping physically impossible
   negative demand to planners.

3. **Monthly grain + exogenous regressors.** Matched the business reality (monthly S&OP cycle) and
   added feedstock price, plant utilization and a REACH-compliance flag as `dynamic_reals` /
   `dynamic_categoricals` in `forecasting_conf_chemicals.yaml`, since demand is driven by those
   external factors as much as by history.

4. **Evidence that survives Git.** Notebook cell outputs are stripped when a repo is exported/zipped,
   which is exactly how this build gets reviewed. So `evidence_runner` writes a plain-Markdown
   `pipeline_run_log.md` with live query results, and `EVIDENCE.md` captures the end-to-end outputs —
   both are reviewable without re-running anything.

## Evidence that it actually ran

All captured live from `e2-demo-field-eng.cloud.databricks.com`, catalog `mmf_demo_aym.m4_advanced`:

- `examples/evidence_runner_executed.md` — the evidence notebook **rendered with its output cells**
  (code + real outputs: backtest leaderboard, scoring counts, Lakebase sync status, a live Genie
  answer). Git-durable twin of the `.ipynb` / `.html` originals, which are included alongside it.
- `examples/EVIDENCE.md` — full end-to-end outputs (data, governance, MLflow leaderboard, Lakebase, Genie).
- `examples/pipeline_run_log.md` — timestamped run log, "every output a live query result."
- MLflow run `83517d0d-584f-43f6-b505-a442fccdf763`; UC tables `monthly_evaluation_output`
  (16,830 rows), `monthly_scoring_output` (1,683 rows).
