# Execution Evidence — Many-Model Demand Forecasting

> All outputs captured live from the Databricks workspace on 2026-10-07.
> Workspace: `e2-demo-field-eng.cloud.databricks.com`
> Catalog: `mmf_demo_aym` · Schema: `m4_advanced`

---

## 1. BASF Chemicals Domain Data (Ingestion + Governance)

Table: `mmf_demo_aym.m4_advanced.basf_chemicals_train`

```
Table: mmf_demo_aym.m4_advanced.basf_chemicals_train
Rows: 720
Products (12): Acronal_DS2373, Acronal_S728, Caprolactam_Grade1, FCC_Catalyst_DMS, Hysorb_B7065, Lupranate_M20S, Lupranate_MI, Pluracol_1509, Pluracol_2090, Ultraform_N2320, Ultramid_A3WG6, Ultramid_B3S
Date range: 2020-01-31 00:00:00 to 2024-12-31 00:00:00
Exogenous regressors: feedstock_price_index=99.31, plant_utilization_rate=0.9394, regulatory_compliance=1

Sample data:
        ds    unique_id       y  feedstock_price_index  plant_utilization_rate  regulatory_compliance
2020-01-31 Ultramid_B3S 2239.99                  99.31                  0.9394                      1
2020-02-29 Ultramid_B3S 2537.67                 104.01                  0.9130                      1
2020-03-31 Ultramid_B3S 2700.26                 113.84                  0.9059                      1
2020-04-30 Ultramid_B3S 2281.68                 111.83                  0.7660                      1
2020-05-31 Ultramid_B3S 2728.29                 107.75                  0.8683                      1
2020-06-30 Ultramid_B3S 2631.59                 114.25                  0.9294                      1
```

Governance: Table and column comments applied via `COMMENT ON TABLE` / `ALTER TABLE ... COMMENT`.
Exogenous regressors configured in `mmf_sa/forecasting_conf_chemicals.yaml` with `dynamic_reals` and `dynamic_categoricals`.

---

## 2. Lakebase Serving Layer

Project: `many-model-forecasting-advanced` (Autoscaling, 2-4 CU)

```
Endpoint: projects/many-model-forecasting-advanced/branches/production/endpoints/primary
Host: ep-noisy-rain-d1p0244n.database.us-west-2.cloud.databricks.com
PostgreSQL: PostgreSQL 17.11 (fcae950) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, 64-bit

Synced tables:
  m4_advanced.lb_m4_monthly_train: 5,940 rows
  m4_advanced.lb_monthly_evaluation_output: 16,830 rows
  m4_advanced.lb_monthly_scoring_output: 1,683 rows

Sample SELECT from Lakebase PostgreSQL:
  Query: SELECT unique_id, model, COUNT(*) FROM m4_advanced.lb_monthly_scoring_output GROUP BY 1,2 ORDER BY 1 LIMIT 8
    M1 | SKTimeProphet                            | 1
    M1 | StatsForecastADIDA                       | 1
    M1 | StatsForecastAutoArima                   | 1
    M1 | StatsForecastAutoCES                     | 1
    M1 | StatsForecastAutoETS                     | 1
    M1 | StatsForecastAutoMfles                   | 1
    M1 | StatsForecastAutoTbats                   | 1
    M1 | StatsForecastAutoTheta                   | 1
  Sync status lb_monthly_evaluation_output: SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE
  Sync status lb_monthly_scoring_output: SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE
  Sync status lb_m4_monthly_train: SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE
```

The Dash app (`apps/app.py`) connects to Lakebase via `apps/lakebase_client.py` for low-latency reads.

---

## 3. MLflow Pipeline Results

```
MLflow Run ID: 83517d0d-584f-43f6-b505-a442fccdf763
Total evaluations: 16,830
Models evaluated: 17
Time series: 99
Backtest windows: 10

  StatsForecastBaselineSeasonalWindowAverage avg_smape=NaN  (990 evals)
  StatsForecastAutoCES                       avg_smape=0.061535  (990 evals)
  StatsForecastAutoETS                       avg_smape=0.062436  (990 evals)
  StatsForecastAutoArima                     avg_smape=0.063498  (990 evals)
  SKTimeProphet                              avg_smape=0.066887  (990 evals)
  StatsForecastAutoTheta                     avg_smape=0.068614  (990 evals)
  StatsForecastBaselineSeasonalNaive         avg_smape=0.084039  (990 evals)
  StatsForecastAutoTbats                     avg_smape=0.084670  (990 evals)
  StatsForecastBaselineWindowAverage         avg_smape=0.086238  (990 evals)
  StatsForecastADIDA                         avg_smape=0.089077  (990 evals)
  StatsForecastIMAPA                         avg_smape=0.089077  (990 evals)
  StatsForecastCrostonOptimized              avg_smape=0.089077  (990 evals)
  StatsForecastTSB                           avg_smape=0.090314  (990 evals)
  StatsForecastAutoMfles                     avg_smape=0.092313  (990 evals)
  StatsForecastCrostonClassic                avg_smape=0.093315  (990 evals)
  StatsForecastBaselineNaive                 avg_smape=0.096275  (990 evals)
  StatsForecastCrostonSBA                    avg_smape=0.106822  (990 evals)

Scoring output — 1683 production forecasts:
  SKTimeProphet                              99 series
  StatsForecastADIDA                         99 series
  StatsForecastAutoArima                     99 series
  StatsForecastAutoCES                       99 series
  StatsForecastAutoETS                       99 series
  StatsForecastAutoMfles                     99 series
  StatsForecastAutoTbats                     99 series
  StatsForecastAutoTheta                     99 series
  StatsForecastBaselineNaive                 99 series
  StatsForecastBaselineSeasonalNaive         99 series
  StatsForecastBaselineSeasonalWindowAverage 99 series
  StatsForecastBaselineWindowAverage         99 series
  StatsForecastCrostonClassic                99 series
  StatsForecastCrostonOptimized              99 series
  StatsForecastCrostonSBA                    99 series
  StatsForecastIMAPA                         99 series
  StatsForecastTSB                           99 series

Best model per time series (first 10):
  M1     -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M10    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M11    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M12    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M13    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M14    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M15    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M16    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M17    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
  M18    -> StatsForecastBaselineSeasonalWindowAverage (smape=None)
```

All 17 models were trained via `mmf_sa/Forecaster.py`, which calls `mlflow.pyfunc.log_model()` with
a registered model name under Unity Catalog. Each model class (`StatsFcForecastingPipeline`,
`NeuralForecastPipeline`, `SKTimeForecastingPipeline`) implements `.fit()` / `.predict()` / `.backtest()`
with results written to the evaluation and scoring output tables above.

---

## 4. Genie Agent — Conversational Query Surface

Genie Space: MMF Demand Forecasting Intelligence
Space ID: 01f1c22c3f621207842ec9d6577c22c7
Tables: monthly_scoring_output, monthly_evaluation_output, m4_monthly_train

--- Conversation 1 ---
Question: Which model has the lowest average sMAPE across all time series?
Status: MessageStatus.COMPLETED

Generated SQL:
WITH model_avg AS (
  SELECT
    `model`,
    AVG(`metric_value`) AS `avg_smape`
  FROM `mmf_demo_aym`.`m4_advanced`.`monthly_evaluation_output`
  WHERE `metric_name` ILIKE '%smape%'
    AND `model` IS NOT NULL
    AND `metric_value` IS NOT NULL
  GROUP BY `model`
), ranked AS (
  SELECT
    `model`,
    `avg_smape`,
    RANK() OVER (ORDER BY `avg_smape` ASC) AS `rnk`
  FROM model_avg
)
SELECT
  `model`,
  `avg_smape`
FROM ranked
WHERE `rnk` <= 1
ORDER BY `avg_smape` ASC, `model` ASC

Answer:
The model with the **lowest average sMAPE** across all time series is **StatsForecastAutoCES**, with an average sMAPE of **0.061535488464333266**. Based on the ranked result, **StatsForecastAutoCES** is the top-performing model on this accuracy measure in the available data.

--- Conversation 2 ---
Question: Compare the top 5 models for time series M1 by average metric value
Status: MessageStatus.COMPLETED

Generated SQL:
WITH model_metric_avg AS ( SELECT `metric_name`, `model`, AVG(`metric_value`) AS `avg_metric_value` FROM `mmf_demo_aym`.`m4_advanced`.`monthly_evaluation_output` WHERE `unique_id` LIKE 'M1%' AND `model` IS NOT NULL AND `metric_name` IS NOT NULL AND `metric_value` IS NOT NULL GROUP BY `metric_name`, `model` ), ranked_models AS ( SELECT `metric_name`, `model`, `avg_metric_value`, RANK() OVER (PARTITION BY `metric_name` ORDER BY `avg_metric_value` DESC) AS `model_rank` FROM model_metric_avg ) SELECT `metric_name`, `model`, `avg_metric_value`, `model_rank` FROM ranked_models WHERE `model_rank` <= 5 ORDER BY `metric_name`, `model_rank`, `model`

Answer:
For time series **M1**, the top 5 models by average metric value for **smape** are ranked from highest to lowest as:
- **StatsForecastCrostonSBA** — **0.0994**
- **StatsForecastCrostonClassic** — **0.0771**
- **StatsForecastTSB** — **0.0742**
- **StatsForecastBaselineNaive** — **0.0741**
- **StatsForecastAutoMfles** — **0.0726**
The leading model, **StatsForecastCrostonSBA**, is noticeably ahead of the other four, which are clustered closely between **0.0726** and **0.0771**.

--- Conversation 3 ---
Question: How many unique time series are in the training data and what is the date range?
Status: MessageStatus.COMPLETED

Generated SQL:
SELECT COUNT(DISTINCT `unique_id`) AS `unique_time_series`, MIN(`ds`) AS `start_date`, MAX(`ds`) AS `end_date`
FROM `mmf_demo_aym`.`m4_advanced`.`m4_monthly_train`
WHERE `unique_id` IS NOT NULL AND `ds` IS NOT NULL

Answer:
The training data contains **99** unique time series, with dates ranging from **2018-01-31** to **2022-12-31**. Based on the full result shown, the training period spans just under **5 years**.

--- Conversation 4 ---
Question: What percentage of models have non-empty prediction intervals in the scoring output?
Status: MessageStatus.COMPLETED

Generated SQL:
SELECT try_divide(100.0 * SUM(CASE WHEN `forecast_lower` IS NOT NULL AND `forecast_upper` IS NOT NULL AND size(`forecast_lower`) > 0 AND size(`forecast_upper`) > 0 THEN 1 ELSE 0 END),COUNT(*)) AS `percentage_with_non_empty_prediction_intervals`
FROM `mmf_demo_aym`.`m4_advanced`.`monthly_scoring_output`

Answer:
**0.00%** of models in `monthly_scoring_output` have non-empty prediction intervals in the scoring output, based on `percentage_with_non_empty_prediction_intervals = 0.00000000000000`. This means **no models** in the result set have both `forecast_lower` and `forecast_upper` populated with non-empty values.



The Genie Agent is a deployed conversational surface over the governed demand data,
distinct from Genie Code (the coding assistant). Business users access it at:
`https://e2-demo-field-eng.cloud.databricks.com/genie/rooms/01f1c22c3f621207842ec9d6577c22c7`

Setup notebook: `examples/setup_genie_agent.ipynb`
Query demo: `examples/genie_demand_queries.ipynb`

---

## 5. Architecture Summary

```
Data Sources --> Lakeflow SDP --> Unity Catalog --> MLflow Registry
                                       |
                                       +---> Lakebase (PostgreSQL 17, synced tables)
                                       |        +---> Dash App (low-latency reads)
                                       |
                                       +---> Genie Agent (natural-language Q&A)
```

| Layer           | Evidence                                                    |
|-----------------|-------------------------------------------------------------|
| Lakeflow        | SDP constructs in `mmf_sa/Forecaster.py`                    |
| Unity Catalog   | 4 tables in `mmf_demo_aym.m4_advanced`, comments, lineage   |
| ML / MLflow     | 17 models, 16,830 evaluations, run ID `83517d0d-584f-...`  |
| Lakebase        | 3 synced tables, 24,453 total rows in PostgreSQL 17.11     |
| Genie Agent     | 4 conversations with real SQL + answers above               |
| App             | Dash app in `apps/app.py` with Lakebase integration         |
