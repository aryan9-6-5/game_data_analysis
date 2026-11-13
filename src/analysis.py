"""Statistical analysis on top of the SQL views. Writes reports/results.json.

Five questions:
  1. Do defect rates differ across platforms?            (chi-square, Kruskal-Wallis, pairwise + Bonferroni)
  2. Is test duration associated with defect discovery?  (Spearman, adjusted logistic regression)
  3. Which defect categories drive release delays?       (blocking-days share + bootstrap CI, OLS)
  4. Which builds are credibly good or bad?              (Wilson confidence intervals)
  5. Can early test data flag builds at risk of failing  (logistic regression / random forest, repeated CV)
     the release-quality threshold?

The data is synthetic, so results show the method working on planted effects.
Run:  python src/analysis.py
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

from db import engine, query_df

OUT = Path(__file__).resolve().parents[1] / "reports"
SEED = 7
EARLY_FRACTION = 0.40  # share of a build's test window used as the "early" signal
warnings.filterwarnings("ignore", category=UserWarning)


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# 1 ---------------------------------------------------------------- platforms
def platform_tests(eng) -> dict:
    ex = query_df("SELECT platform_name, status FROM v_execution_facts WHERE status <> 'Blocked'", eng)
    table = pd.crosstab(ex["platform_name"], ex["status"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))

    pairs, names = [], table.index.tolist()
    for a, b in itertools.combinations(names, 2):
        sub = table.loc[[a, b]]
        c2, pp, _, _ = stats.chi2_contingency(sub)
        odds = (sub.loc[a, "Failed"] / sub.loc[a, "Passed"]) / (sub.loc[b, "Failed"] / sub.loc[b, "Passed"])
        se = np.sqrt((1 / sub.values).sum())
        lo, hi = np.exp(np.log(odds) + np.array([-1.96, 1.96]) * se)
        pairs.append(dict(a=a, b=b, odds_ratio_fail=r(odds, 3), ci95=[r(lo, 3), r(hi, 3)],
                          p_bonferroni=r(min(1.0, pp * 3), 6)))

    per_build = query_df("""
        SELECT e.build_id, p.platform_name,
               100.0 * COUNT(d.defect_id) / COUNT(DISTINCT e.execution_id) AS rate
        FROM test_executions e JOIN platforms p USING (platform_id)
        LEFT JOIN defects d ON d.build_id = e.build_id AND d.platform_id = e.platform_id AND d.execution_id = e.execution_id
        GROUP BY 1, 2""", eng)
    groups = [g["rate"].to_numpy() for _, g in per_build.groupby("platform_name")]
    h, kp = stats.kruskal(*groups)
    return dict(
        fail_rate_by_platform={k: r(v, 4) for k, v in (table["Failed"] / table.sum(axis=1)).items()},
        chi2=r(chi2, 2), dof=int(dof), p_value=float(p), cramers_v=r(cramers_v, 4),
        pairwise=pairs, kruskal_h=r(h, 2), kruskal_p=float(kp),
        note="Execution-level chi-square on Passed/Failed; Kruskal-Wallis on per-build failed-run rates.",
    )


# 2 ------------------------------------------------------------------ duration
def duration_tests(eng) -> dict:
    ex = query_df("""SELECT status, duration_min, found_defect, area, platform_name, experience_years
                     FROM v_execution_facts WHERE status <> 'Blocked'""", eng)
    ex["failed"] = (ex["status"] == "Failed").astype(int)
    ex["log_dur"] = np.log2(ex["duration_min"])
    rho, p = stats.spearmanr(ex["duration_min"], ex["failed"])

    within = []
    for area, g in ex.groupby("area"):
        rr, pp = stats.spearmanr(g["duration_min"], g["failed"])
        within.append(dict(area=area, spearman=r(rr), p=float(pp)))

    # Unadjusted vs adjusted: area drives both duration and failure, so pooled correlation is confounded.
    m = smf.logit("failed ~ log_dur + C(area) + C(platform_name) + experience_years", ex).fit(disp=0)
    or_ = np.exp(m.params["log_dur"]); lo, hi = np.exp(m.conf_int().loc["log_dur"])

    pb = query_df("""SELECT m.build_id, m.defects_per_100_exec,
                            SUM(e.duration_min) / COUNT(*) AS avg_duration
                     FROM v_build_metrics m JOIN test_executions e USING (build_id) GROUP BY 1, 2""", eng)
    brho, bp = stats.spearmanr(pb["avg_duration"], pb["defects_per_100_exec"])
    return dict(
        pooled_spearman=r(rho), pooled_p=float(p), within_area=within,
        adjusted_odds_ratio_per_doubling=r(or_, 3), adjusted_ci95=[r(lo, 3), r(hi, 3)],
        adjusted_p=float(m.pvalues["log_dur"]),
        build_level_spearman=r(brho), build_level_p=float(bp),
        note="Odds ratio is the change in odds of a failed run per doubling of duration, adjusted for area, platform, tester experience.",
    )


# 3 ------------------------------------------------------------ release delays
def delay_drivers(eng) -> dict:
    d = query_df("""SELECT category, EXTRACT(EPOCH FROM resolved_at - opened_at) / 86400.0 AS days
                    FROM defects WHERE blocks_release AND resolved_at IS NOT NULL""", eng)
    total = d["days"].sum()
    share = (d.groupby("category")["days"].sum() / total).sort_values(ascending=False)
    rng = np.random.default_rng(SEED)
    boots = {c: [] for c in share.index}
    for _ in range(1000):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        t = s["days"].sum()
        g = s.groupby("category")["days"].sum() / t
        for c in boots:
            boots[c].append(g.get(c, 0.0))
    pareto = [dict(category=c, share=r(share[c]), ci95=[r(np.percentile(boots[c], 2.5)), r(np.percentile(boots[c], 97.5))])
              for c in share.index]

    cnt = query_df("""SELECT b.build_id, b.release_delay_days AS delay, d.category, COUNT(*) AS n
                      FROM builds b JOIN defects d USING (build_id) WHERE d.blocks_release
                      GROUP BY 1, 2, 3""", eng)
    wide = cnt.pivot_table(index=["build_id", "delay"], columns="category", values="n", fill_value=0).reset_index()
    wide.columns = [str(c).replace("/", "_").replace(" ", "_") for c in wide.columns]
    cats = [c for c in wide.columns if c not in ("build_id", "delay")]
    ols = smf.ols("delay ~ " + " + ".join(cats), wide).fit(cov_type="HC3")
    coefs = [dict(category=c, extra_delay_days_per_blocking_defect=r(ols.params[c], 3),
                  ci95=[r(ols.conf_int().loc[c, 0], 3), r(ols.conf_int().loc[c, 1], 3)], p=r(ols.pvalues[c], 4))
             for c in cats]
    coefs.sort(key=lambda x: -(x["extra_delay_days_per_blocking_defect"] or 0))
    return dict(pareto_blocking_days=pareto, ols_coefficients=coefs, ols_r2=r(ols.rsquared, 3), n_builds=int(len(wide)),
                note="Blocking defect = Critical, or Major found in an RC build. Share = fraction of blocking defect-days.")


# 4 ------------------------------------------------------------ build quality CIs
def build_cis(eng) -> dict:
    m = query_df("SELECT build_id, build_type, passed, failed, pass_rate, planned_release_date "
                 "FROM v_build_metrics ORDER BY planned_release_date", eng)
    thr = float(query_df("SELECT min_pass_rate FROM v_thresholds", eng).iloc[0, 0])
    n = m["passed"] + m["failed"]
    lo, hi = proportion_confint(m["passed"], n, alpha=0.05, method="wilson")
    m["ci_lo"], m["ci_hi"] = lo, hi
    m["verdict"] = np.where(m["ci_hi"] < thr, "credibly below threshold",
                    np.where(m["ci_lo"] >= thr, "credibly above threshold", "inconclusive"))
    by_type = []
    for t, g in m.groupby("build_type"):
        k, nn = g["passed"].sum(), (g["passed"] + g["failed"]).sum()
        a, b = proportion_confint(k, nn, method="wilson")
        by_type.append(dict(build_type=t, pass_rate=r(k / nn), ci95=[r(a), r(b)]))
    rc, al = m[m.build_type == "RC"], m[m.build_type == "Alpha"]
    p1, n1 = rc["passed"].sum() / (rc["passed"] + rc["failed"]).sum(), (rc["passed"] + rc["failed"]).sum()
    p2, n2 = al["passed"].sum() / (al["passed"] + al["failed"]).sum(), (al["passed"] + al["failed"]).sum()
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = dict(rc_minus_alpha=r(p1 - p2), ci95=[r(p1 - p2 - 1.96 * se), r(p1 - p2 + 1.96 * se)])
    builds = m[["build_id", "build_type", "pass_rate", "ci_lo", "ci_hi", "verdict"]].round(4).to_dict("records")
    return dict(threshold=thr, builds=builds, by_type=by_type, rc_vs_alpha=diff,
                verdict_counts=m["verdict"].value_counts().to_dict(),
                note="Wilson 95% intervals on each build's pass rate; a build is only 'credibly' above/below when the whole interval is.")


# 5 ------------------------------------------------------------ risk model
def risk_model(eng) -> dict:
    ex = query_df("""SELECT e.build_id, e.executed_at, e.status, e.duration_min, e.platform_id
                     FROM test_executions e""", eng)
    df = query_df("SELECT build_id, opened_at, severity FROM defects", eng)
    bm = query_df("""SELECT build_id, build_type, change_size, meets_release_quality, planned_release_date
                     FROM v_build_metrics ORDER BY planned_release_date""", eng)
    rows = []
    for b in bm.itertuples():
        e = ex[ex.build_id == b.build_id].sort_values("executed_at")
        cut = e["executed_at"].min() + (e["executed_at"].max() - e["executed_at"].min()) * EARLY_FRACTION
        early = e[e["executed_at"] <= cut]
        done = early[early.status != "Blocked"]
        d = df[(df.build_id == b.build_id) & (df.opened_at <= cut)]
        rows.append(dict(
            build_id=b.build_id,
            early_pass_rate=(done.status == "Passed").mean(),
            early_defects_per_100=100 * len(d) / len(early),
            early_critical=(d.severity == "Critical").sum(),
            early_avg_duration=early.duration_min.mean(),
            early_pc_share=(early.platform_id == 1).mean(),
            log_change_size=np.log(b.change_size),
            is_alpha=int(b.build_type == "Alpha"), is_rc=int(b.build_type == "RC"),
            at_risk=int(not b.meets_release_quality)))
    f = pd.DataFrame(rows)
    f["prev_build_at_risk"] = f["at_risk"].shift(1)  # known before testing starts
    f = f.dropna().reset_index(drop=True)
    feats = [c for c in f.columns if c not in ("build_id", "at_risk")]
    X, y = f[feats], f["at_risk"]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED),
    }
    res = {}
    for name, mod in models.items():
        s = cross_val_score(mod, X, y, cv=cv, scoring="roc_auc")
        res[name] = dict(auc_mean=r(s.mean(), 3), auc_sd=r(s.std(), 3),
                         auc_p5_p95=[r(np.percentile(s, 5), 3), r(np.percentile(s, 95), 3)])
    base = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                           X[["prev_build_at_risk"]], y, cv=cv, scoring="roc_auc")
    res["Baseline: previous build only"] = dict(auc_mean=r(base.mean(), 3), auc_sd=r(base.std(), 3))

    split = int(len(f) * 0.7)  # time-ordered holdout: train on earlier builds, test on later ones
    lr = models["Logistic regression"].fit(X[:split], y[:split])
    holdout = roc_auc_score(y[split:], lr.predict_proba(X[split:])[:, 1]) if y[split:].nunique() > 1 else None
    lr.fit(X, y)
    coef = pd.Series(lr[-1].coef_[0], index=feats).sort_values(key=abs, ascending=False)
    lr_all = lr.predict_proba(X)[:, 1]
    f["risk_score"] = lr_all
    return dict(
        n_builds=int(len(f)), at_risk_share=r(y.mean(), 3), early_fraction=EARLY_FRACTION,
        cv=res, time_holdout_auc=r(holdout, 3), holdout_n=int(len(f) - split),
        standardized_coefficients={k: r(v, 3) for k, v in coef.items()},
        build_risk=f[["build_id", "risk_score", "at_risk"]].round(3).to_dict("records"),
        note=("At-risk = fails the release-quality threshold on the full test cycle. Features use only the first "
              f"{int(EARLY_FRACTION*100)}% of test time plus planning-time facts; the early runs are part of the "
              "full-cycle outcome, so this is an early-warning score, not an independent forecast."),
    )


def main() -> None:
    eng = engine()
    results = dict(
        data_note="SYNTHETIC data generated by src/generate_data.py. Results demonstrate method, not real findings.",
        platform=platform_tests(eng), duration=duration_tests(eng), delay=delay_drivers(eng),
        build_ci=build_cis(eng), risk=risk_model(eng))
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    p, d, dl, ci, rk = (results[k] for k in ("platform", "duration", "delay", "build_ci", "risk"))
    print("PLATFORM chi2=%s p=%.2e V=%s | KW p=%.2e" % (p["chi2"], p["p_value"], p["cramers_v"], p["kruskal_p"]))
    print("  fail rates", p["fail_rate_by_platform"])
    for x in p["pairwise"]: print("  ", x)
    print("DURATION pooled rho=%s | within-area %s" % (d["pooled_spearman"], [(w["area"], w["spearman"]) for w in d["within_area"]]))
    print("  adj OR/doubling", d["adjusted_odds_ratio_per_doubling"], d["adjusted_ci95"], "p=%.2e" % d["adjusted_p"], "| build-level rho", d["build_level_spearman"])
    print("DELAY pareto", [(x["category"], x["share"], x["ci95"]) for x in dl["pareto_blocking_days"]])
    print("  OLS r2", dl["ols_r2"], [(x["category"], x["extra_delay_days_per_blocking_defect"], x["p"]) for x in dl["ols_coefficients"]])
    print("BUILD CI", ci["verdict_counts"], ci["by_type"], ci["rc_vs_alpha"])
    print("RISK n=%d at-risk=%s" % (rk["n_builds"], rk["at_risk_share"]), rk["cv"], "holdout", rk["time_holdout_auc"], "n", rk["holdout_n"])
    print("  coefs", rk["standardized_coefficients"])


if __name__ == "__main__":
    main()
