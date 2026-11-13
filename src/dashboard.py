"""Build the self-contained HTML dashboard and the Power BI import files.

Reads the SQL views and reports/results.json. Run after load_db.py and analysis.py.
Run:  python src/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db import engine, query_df

ROOT = Path(__file__).resolve().parents[1]
PBI = ROOT / "powerbi"
BLUE, ORANGE, RED, GREY, GREEN = "#2a6fb0", "#e08a1e", "#c8412f", "#8a8f98", "#3a9d6a"
VIEWS = ["v_build_trend", "v_platform_metrics", "v_category_metrics", "v_tester_efficiency"]


def kpis(eng) -> dict:
    return query_df("""
        SELECT ROUND(100.0 * SUM(passed) / SUM(passed + failed), 1) AS pass_rate,
               (SELECT COUNT(*) FROM defects WHERE status = 'Open')            AS open_defects,
               (SELECT ROUND(AVG(EXTRACT(EPOCH FROM resolved_at - opened_at) / 86400.0)::numeric, 1)
                  FROM defects WHERE resolved_at IS NOT NULL)                  AS avg_res_days,
               (SELECT COUNT(*) FROM defects WHERE severity = 'Critical')      AS critical,
               (SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE reopen_count > 0)
                       / COUNT(*) FILTER (WHERE resolved_at IS NOT NULL), 1) FROM defects) AS reopen_rate,
               (SELECT COUNT(*) FILTER (WHERE meets_release_quality) FROM v_build_metrics) AS ready,
               (SELECT COUNT(*) FROM v_build_metrics)                          AS builds
        FROM v_build_metrics""", eng).iloc[0].to_dict()


def tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div>'
            f'<div class="s">{sub}</div></div>')


def layout(fig, title, h=360):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)), height=h,
                      margin=dict(l=50, r=20, t=50, b=40), template="plotly_white",
                      font=dict(family="Segoe UI, Helvetica, Arial", size=12), legend=dict(orientation="h", y=-0.2))
    return fig


def main() -> None:
    eng = engine()
    res = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    k = kpis(eng)
    bt = query_df("SELECT * FROM v_build_trend ORDER BY planned_release_date", eng)
    ci = pd.DataFrame(res["build_ci"]["builds"]).set_index("build_id").loc[bt["build_id"]].reset_index()
    thr = res["build_ci"]["threshold"]

    # 1 build trend with Wilson CI band
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=ci.build_id, y=ci.ci_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=ci.build_id, y=ci.ci_lo, fill="tonexty", fillcolor="rgba(42,111,176,.18)",
                            line=dict(width=0), name="95% CI (Wilson)", hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=bt.build_id, y=bt.pass_rate, mode="lines+markers", name="Pass rate",
                            line=dict(color=BLUE, width=2), marker=dict(size=5)))
    f1.add_trace(go.Scatter(x=bt.build_id, y=bt.pass_rate_roll5, name="5-build average", line=dict(color=ORANGE, dash="dot")))
    f1.add_hline(y=thr, line=dict(color=RED, dash="dash"), annotation_text=f"threshold {thr:.1%}")
    f1.update_yaxes(tickformat=".0%"); f1.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    layout(f1, "Pass rate by build, with 95% confidence band", 380)

    # 2 critical + defect rate
    f2 = make_subplots(specs=[[{"secondary_y": True}]])
    f2.add_bar(x=bt.build_id, y=bt.critical_defects, name="Critical defects",
               marker_color=[RED if v > 4 else GREY for v in bt.critical_defects])
    f2.add_scatter(x=bt.build_id, y=bt.defects_per_100_exec, name="Defects / 100 runs",
                   line=dict(color=BLUE), secondary_y=True)
    f2.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    layout(f2, "Critical defects per build (red = over limit) and discovery rate", 380)

    # 3 platform
    pm = query_df("SELECT * FROM v_platform_metrics ORDER BY defects_per_100_exec DESC", eng)
    f3 = go.Figure(go.Bar(x=pm.platform_name, y=pm.defects_per_100_exec, marker_color=[RED, GREY, GREY],
                          text=pm.defects_per_100_exec.round(2), textposition="outside"))
    f3.update_yaxes(range=[0, pm.defects_per_100_exec.max() * 1.2])
    p = res["platform"]
    layout(f3, f"Defects per 100 runs by platform (chi-square p={p['p_value']:.1e}, Cramér's V={p['cramers_v']})", 340)

    # 4 category Pareto
    par = pd.DataFrame(res["delay"]["pareto_blocking_days"])
    f4 = go.Figure(go.Bar(x=par.category, y=par.share, error_y=dict(type="data", symmetric=False,
                   array=[b - a for a, b in zip(par.share, par.ci95.str[1])],
                   arrayminus=[a - b for a, b in zip(par.share, par.ci95.str[0])]), marker_color=BLUE))
    f4.update_yaxes(tickformat=".0%"); f4.update_xaxes(tickangle=-30)
    layout(f4, "Share of release-blocking defect-days by category (95% bootstrap CI)", 340)

    # 5 tester efficiency
    te = query_df("SELECT * FROM v_tester_efficiency", eng)
    order = ["<2y", "2-5y", "5y+"]
    f5 = go.Figure()
    for team, g in te.groupby("team"):
        g = g.set_index("experience_band").reindex(order)
        f5.add_bar(x=order, y=g.executions_per_hour, name=team)
    f5.update_layout(barmode="group")
    layout(f5, "Test execution efficiency (runs per tester-hour) by team and experience", 340)

    # 6 risk model
    rk = pd.DataFrame(res["risk"]["build_risk"])
    f6 = go.Figure(go.Bar(x=rk.build_id, y=rk.risk_score,
                          marker_color=[RED if a else GREEN for a in rk.at_risk]))
    cv = res["risk"]["cv"]["Logistic regression"]
    f6.update_yaxes(tickformat=".0%", range=[0, 1]); f6.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    layout(f6, f"Early-warning risk score (red = actually failed threshold; CV AUC {cv['auc_mean']}±{cv['auc_sd']})", 340)

    figs = [f1, f2, f3, f4, f5, f6]
    div = lambda i, f: f.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False))
    body = "".join(f'<div class="card{" wide" if i < 2 else ""}">{div(i, f)}</div>' for i, f in enumerate(figs))
    tiles = "".join([
        tile("Test pass rate", f"{k['pass_rate']}%", "Passed / (Passed + Failed)"),
        tile("Open defects", f"{int(k['open_defects'])}", "incl. deferred minor/trivial"),
        tile("Avg resolution time", f"{k['avg_res_days']} days", "resolved defects only"),
        tile("Critical defects", f"{int(k['critical'])}", f"{int(k['ready'])}/{int(k['builds'])} builds release-ready"),
        tile("Defect reopen rate", f"{k['reopen_rate']}%", "of resolved defects"),
    ])
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Game Quality and Release Analytics</title><style>
body{{margin:0;background:#f4f6f9;font-family:Segoe UI,Helvetica,Arial,sans-serif;color:#1d2530}}
header{{padding:18px 28px;background:#1b2a3a;color:#fff}} h1{{margin:0;font-size:20px}}
.banner{{background:#fff4d6;border-bottom:1px solid #e8d49a;padding:8px 28px;font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;padding:16px 28px 4px}}
.tile{{background:#fff;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.v{{font-size:28px;font-weight:600}} .l{{font-size:13px;margin-top:2px}} .s{{font-size:11px;color:#6a7480;margin-top:2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:12px;padding:12px 28px 28px}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:6px;overflow:hidden}}
.wide{{grid-column:1/-1}}</style></head><body>
<header><h1>Game Quality and Release Analytics</h1></header>
<div class="banner"><b>Synthetic data.</b> 60 simulated builds of a fictional title, generated by src/generate_data.py. Not from any real studio or game.</div>
<div class="tiles">{tiles}</div><div class="grid">{body}</div></body></html>"""
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dashboard.html").write_text(html, encoding="utf-8")

    # Power BI import files + measures
    PBI.mkdir(exist_ok=True)
    for v in VIEWS:
        query_df(f"SELECT * FROM {v}", eng).to_csv(PBI / f"{v}.csv", index=False)
    pd.DataFrame(res["build_ci"]["builds"]).to_csv(PBI / "build_confidence_intervals.csv", index=False)
    pd.DataFrame(res["risk"]["build_risk"]).to_csv(PBI / "build_risk_scores.csv", index=False)
    print("wrote reports/dashboard.html and", len(list(PBI.glob("*.csv"))), "CSVs in powerbi/")


if __name__ == "__main__":
    main()
