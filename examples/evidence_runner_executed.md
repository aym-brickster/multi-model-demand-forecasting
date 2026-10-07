# Executed Evidence Notebook (code + live outputs)

> Rendered from a live Databricks **serverless** run (`e2-demo-field-eng`), job run `617471733943740`, 2026-10-07. Each code cell is followed by its real, unmodified output. This is the Git-durable twin of `evidence_runner_executed.ipynb` / `.html`.


# Execution Evidence — Executed Notebook

> Live run on Databricks **serverless** (`e2-demo-field-eng`), exported from job run `617471733943740` on 2026-10-07. Every output cell below is a real, unmodified result — backtest leaderboard, MLflow run id, production scoring, Lakebase PostgreSQL sync status, and a live Genie answer.


# Execution Evidence — Many-Model Demand Forecasting Pipeline

All outputs below are live-executed evidence from the Databricks workspace.
Catalog: `mmf_demo_aym` · Schema: `m4_advanced`


```python
%pip install "psycopg[binary]>=3.0" -q
```

**Output:**
```
[43mNote: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.[0m
```


```python
df = spark.table("mmf_demo_aym.m4_advanced.basf_chemicals_train")
print(f"Table: mmf_demo_aym.m4_advanced.basf_chemicals_train")
print(f"Total rows: {df.count()}")
print(f"Products: {df.select('unique_id').distinct().count()}")
date_range = df.selectExpr("MIN(ds) as min_ds", "MAX(ds) as max_ds").collect()[0]
print(f"Date range: {date_range.min_ds} to {date_range.max_ds}")
print(f"\nExogenous regressors: feedstock_price_index, plant_utilization_rate, regulatory_compliance")
print(f"\nSample data:")
display(df.limit(12))
```

**Output:**
```
Table: mmf_demo_aym.m4_advanced.basf_chemicals_train
Total rows: 720
Products: 12
Date range: 2020-01-31 00:00:00 to 2025-03-31 00:00:00

Exogenous regressors: feedstock_price_index, plant_utilization_rate, regulatory_compliance

Sample data:
ds                        unique_id     y        feedstock_price_index  plant_utilization_rate  regulatory_compliance
2020-01-31T00:00:00.000Z  Ultramid_B3S  2193.52  99.14                  0.8755                  1                    
2020-02-29T00:00:00.000Z  Ultramid_B3S  2278.79  100.49                 0.8477                  1                    
2020-03-31T00:00:00.000Z  Ultramid_B3S  2272.58  110.58                 0.9326                  0                    
2020-04-30T00:00:00.000Z  Ultramid_B3S  2348.96  116.43                 0.815                   1                    
2020-05-31T00:00:00.000Z  Ultramid_B3S  2297.83  123.88                 0.8384                  1                    
2020-06-30T00:00:00.000Z  Ultramid_B3S  2251.22  117.86                 0.6821                  1                    
2020-07-31T00:00:00.000Z  Ultramid_B3S  2252.5   118.8                  0.8951                  1                    
2020-08-31T00:00:00.000Z  Ultramid_B3S  2181.57  112.11                 0.8793                  1                    
2020-09-30T00:00:00.000Z  Ultramid_B3S  2021.76  114.29                 0.8867                  1                    
2020-10-31T00:00:00.000Z  Ultramid_B3S  2169.22  106.83                 0.7061                  1                    
2020-11-30T00:00:00.000Z  Ultramid_B3S  2140.18  108.53                 0.8413                  0                    
2020-12-31T00:00:00.000Z  Ultramid_B3S  2216.04  105.43                 0.735                   1
```


```python
eval_results = spark.sql("""
    SELECT model, run_id,
        COUNT(DISTINCT unique_id) as series_evaluated,
        COUNT(DISTINCT backtest_window_start_date) as backtest_windows,
        COUNT(*) as total_evaluations,
        ROUND(AVG(metric_value), 6) as avg_smape
    FROM mmf_demo_aym.m4_advanced.monthly_evaluation_output
    GROUP BY model, run_id ORDER BY avg_smape ASC
""")
total = eval_results.selectExpr("SUM(total_evaluations)").collect()[0][0]
print(f"MLflow Run ID: {eval_results.collect()[0].run_id}")
print(f"Total evaluations: {int(total):,}")
print(f"Models: {eval_results.count()}, Time series: 99, Backtest windows: 10")
display(eval_results)
```

**Output:**
```
MLflow Run ID: 83517d0d-584f-43f6-b505-a442fccdf763
Total evaluations: 16,830
Models: 17, Time series: 99, Backtest windows: 10
model                                       run_id                                series_evaluated  backtest_windows  total_evaluations  avg_smape
StatsForecastBaselineSeasonalWindowAverage  83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                         
StatsForecastAutoCES                        83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.061535 
StatsForecastAutoETS                        83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.062436 
StatsForecastAutoArima                      83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.063498 
SKTimeProphet                               83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.066887 
StatsForecastAutoTheta                      83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.068614 
StatsForecastBaselineSeasonalNaive          83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.084039 
StatsForecastAutoTbats                      83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.08467  
StatsForecastBaselineWindowAverage          83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.086238 
StatsForecastCrostonOptimized               83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.089077 
StatsForecastADIDA                          83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.089077 
StatsForecastIMAPA                          83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.089077 
StatsForecastTSB                            83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.090314 
StatsForecastAutoMfles                      83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.092313 
StatsForecastCrostonClassic                 83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.093315 
StatsForecastBaselineNaive                  83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.096275 
StatsForecastCrostonSBA                     83517d0d-584f-43f6-b505-a442fccdf763  99                10                990                0.106822
```


```python
scoring = spark.sql("""
    SELECT model, COUNT(DISTINCT unique_id) as series_scored, run_id
    FROM mmf_demo_aym.m4_advanced.monthly_scoring_output
    GROUP BY model, run_id ORDER BY model
""")
total_scored = scoring.selectExpr("SUM(series_scored)").collect()[0][0]
print(f"Total production forecasts: {int(total_scored):,} across {scoring.count()} models")
display(scoring)
```

**Output:**
```
Total production forecasts: 1,683 across 17 models
model                                       series_scored  run_id                              
SKTimeProphet                               99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastADIDA                          99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastAutoArima                      99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastAutoCES                        99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastAutoETS                        99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastAutoMfles                      99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastAutoTbats                      99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastAutoTheta                      99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastBaselineNaive                  99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastBaselineSeasonalNaive          99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastBaselineSeasonalWindowAverage  99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastBaselineWindowAverage          99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastCrostonClassic                 99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastCrostonOptimized               99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastCrostonSBA                     99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastIMAPA                          99             83517d0d-584f-43f6-b505-a442fccdf763
StatsForecastTSB                            99             83517d0d-584f-43f6-b505-a442fccdf763
```


```python
import psycopg
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
endpoints = list(w.postgres.list_endpoints(parent="projects/many-model-forecasting-advanced/branches/production"))
ep = endpoints[0]
host = ep.status.hosts.host
cred = w.postgres.generate_database_credential(endpoint=ep.name)
username = w.current_user.me().user_name
print(f"Lakebase Autoscaling Project: many-model-forecasting-advanced")
print(f"Endpoint: {ep.name}")
print(f"Host: {host}")
with psycopg.connect(host=host, dbname="databricks_postgres", user=username, password=cred.token, sslmode="require") as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        print(f"Connected: {cur.fetchone()[0]}")
        print(f"\nSynced tables in Lakebase PostgreSQL:")
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'm4_advanced' ORDER BY tablename")
        for r in cur.fetchall():
            cur.execute(f'SELECT COUNT(*) FROM m4_advanced.\"{r[0]}\"')
            print(f"  m4_advanced.{r[0]}: {cur.fetchone()[0]:,} rows")
        print(f"\nSample SELECT — scoring forecasts per model:")
        cur.execute('SELECT unique_id, model, COUNT(*) as n FROM m4_advanced."lb_monthly_scoring_output" GROUP BY 1,2 ORDER BY 1 LIMIT 8')
        for r in cur.fetchall():
            print(f"  {r[0]:>4} | {r[1]:<40} | {r[2]}")
for tbl in ["lb_monthly_evaluation_output", "lb_monthly_scoring_output", "lb_m4_monthly_train"]:
    st = w.postgres.get_synced_table(name=f"synced_tables/mmf_demo_aym.m4_advanced.{tbl}")
    print(f"  Sync status {tbl}: {st.status.detailed_state}")
```

**Output:**
```
Lakebase Autoscaling Project: many-model-forecasting-advanced
Endpoint: projects/many-model-forecasting-advanced/branches/production/endpoints/primary
Host: ep-noisy-rain-d1p0244n.database.us-west-2.cloud.databricks.com
Connected: PostgreSQL 17.11 (fcae950) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, 64-bit

Synced tables in Lakebase PostgreSQL:
  m4_advanced.lb_m4_monthly_train: 5,940 rows
  m4_advanced.lb_monthly_evaluation_output: 16,830 rows
  m4_advanced.lb_monthly_scoring_output: 1,683 rows

Sample SELECT — scoring forecasts per model:
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


```python
import time
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
GENIE_SPACE_ID = "01f1c22c3f621207842ec9d6577c22c7"
print(f"Genie Space: MMF Demand Forecasting Intelligence")
print(f"Space ID: {GENIE_SPACE_ID}")
print(f"URL: https://{w.config.host}/genie/rooms/{GENIE_SPACE_ID}")
print(f"\nAsking: 'Which model has the lowest average sMAPE across all time series?'")
conv = w.genie.start_conversation(space_id=GENIE_SPACE_ID, content="Which model has the lowest average sMAPE across all time series?")
for _ in range(60):
    msg = w.genie.get_message(space_id=GENIE_SPACE_ID, conversation_id=conv.conversation_id, message_id=conv.message_id)
    if msg.status in ("COMPLETED", "FAILED"):
        break
    time.sleep(2)
print(f"Status: {msg.status}")
if hasattr(msg, 'attachments') and msg.attachments:
    for att in msg.attachments:
        if hasattr(att, 'query') and att.query:
            print(f"\nGenerated SQL:\n{att.query.query}")
        if hasattr(att, 'text') and att.text:
            print(f"\nAnswer:\n{att.text.content}")
```

**Output:**
```
Genie Space: MMF Demand Forecasting Intelligence
Space ID: 01f1c22c3f621207842ec9d6577c22c7
URL: https://https://e2-demo-field-eng.cloud.databricks.com/genie/rooms/01f1c22c3f621207842ec9d6577c22c7

Asking: 'Which model has the lowest average sMAPE across all time series?'
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
The model with the **lowest average sMAPE** is **StatsForecastAutoCES**, with an average sMAPE of **0.061535488464333266** across all time series. Based on the result returned, **StatsForecastAutoCES** is the top-performing model on this measure.
```


## Summary
| Layer | Evidence |
|-------|--------|
| **BASF Data** | 720 rows, 12 product families, 3 exogenous regressors |
| **MLflow** | 17 models, 16,830 evaluations, run ID above |
| **Lakebase** | 3 synced tables in PostgreSQL 17, all ONLINE |
| **Genie Agent** | Live conversation transcript with generated SQL + answer |
| **App** | Dash app in `apps/app.py` with Lakebase integration |
