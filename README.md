# Game Quality and Release Analytics

An end-to-end analytics project for a game quality-verification team: PostgreSQL metric layer, statistical testing, a release-risk model and a dashboard that answers *"is this build ready to release?"*

> **All data is synthetic.** `src/generate_data.py` simulates 60 builds of a fictional title (36 testers, ~36k test executions, ~2.2k defects). It is not from any real studio or game. The generator plants known effects (PC fails more, bigger builds fail more, online bugs take longer to fix, slow blocking fixes delay releases), so the analysis shows that the **method** recovers real structure. The numbers are not findings about real games.

## Pipeline

```
generate_data.py -> data/raw/*.csv -> load_db.py -> PostgreSQL (tables + views)
                                                       |-> analysis.py  -> reports/results.json
                                                       |-> dashboard.py -> reports/dashboard.html + powerbi/*.csv
```

| Piece | Where |
|---|---|
| Schema with constraints, indexes | `sql/01_schema.sql` |
| Metric views (CTEs, window functions, `FILTER`) | `sql/02_views.sql` |
| Statistics and model | `src/analysis.py` |
| Dashboard (self-contained HTML) | `reports/dashboard.html` |
| Power BI import CSVs and DAX measures | `powerbi/` |
| Data-integrity tests (7) | `tests/` |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # set PGPASSWORD; port 5440 matches docker-compose.yml
docker compose up -d
python src/generate_data.py
python src/load_db.py
python src/analysis.py
python src/dashboard.py
python -m pytest tests -q
```

## Metrics (all computed in SQL)

Test pass rate (Passed / (Passed + Failed); Blocked reported separately), defect discovery rate (per 100 runs), critical defects per build, defect reopening rate, average resolution time, test execution efficiency (runs and defects per tester-hour), build-over-build trend with 5-build rolling averages, platform-wise defect distribution. A build is **release-ready** when pass rate >= 92.5% and critical defects <= 4 (one place to change: `v_thresholds`).

## Statistical analysis and what it shows (on this synthetic data)

1. **Platform differences.** Failed-run rate: PC 8.1%, Xbox 6.0%, PlayStation 5.2%. Chi-square p < 1e-20, but **Cramér's V is only 0.05**: highly significant because n is large, small in effect size. PC vs PlayStation odds ratio 1.62 (95% CI 1.46 to 1.80). A per-build Kruskal-Wallis test agrees (p = 1.7e-4).
2. **Duration vs discovery.** The pooled correlation is weak (Spearman 0.07). After adjusting for test area, platform and tester experience, each doubling of run time raises the odds of a failed run by about 16% (OR 1.16, CI 1.10 to 1.23). At build level the correlation is slightly *negative* (-0.26), so time spent per run is not a reliable build-level predictor of defect volume.
3. **Release-delay drivers.** Crash/Stability (27%) and Online/Network (22%) account for about half of all release-blocking defect-days, with bootstrap CIs. A per-build OLS of delay on blocking-defect counts (R² 0.74) is noisy with 60 builds, for example Audio gets a large coefficient from very few cases, so the Pareto share is the more trustworthy view.
4. **Build-quality intervals.** Wilson 95% intervals classify 21 builds as credibly above the pass-rate threshold, 4 as credibly below, and **35 as inconclusive**. A single point estimate would have hidden that uncertainty. RC builds pass 3.6 points more often than Alpha builds (CI 2.9 to 4.3).
5. **Risk model.** Features use only the first 40% of a build's test window plus planning facts (change size, build type, previous build). Repeated 5-fold CV (20 repeats, 59 builds): logistic regression AUC **0.85 ± 0.10**, random forest 0.87 ± 0.10, previous-build-only baseline 0.59. A time-ordered holdout (last 18 builds) gives 0.83. Intervals are wide because the sample is small. The early runs are part of the final outcome, so treat the score as an early-warning signal, not an independent forecast.

## Limits

- Synthetic data with planted effects: it validates the pipeline, not real-world conclusions.
- 60 builds is small for modelling; expect wide uncertainty.
- `powerbi/measures.dax` has not been opened in Power BI Desktop; build and check the `.pbix` there. The HTML dashboard is the verified deliverable.
