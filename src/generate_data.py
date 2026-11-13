"""Generate a SYNTHETIC game-QA dataset (builds, testers, test executions, defects).

All data is simulated. The fictional title is "Pitch Legends"; nothing here comes
from any real studio. The simulation bakes in plausible mechanisms (platform
differences, bigger change sets -> more failures, longer fixes for online bugs,
release delays caused by slow-to-fix blocking defects). The analysis scripts then
try to recover those mechanisms, so findings describe the *method*, not real games.

Run:  python src/generate_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BUILDS = 60
EXEC_PER_BUILD = 600
TEST_WINDOW_DAYS = 10
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

PLATFORMS = pd.DataFrame(
    {"platform_id": [1, 2, 3], "platform_name": ["PC", "PlayStation", "Xbox"]}
)
PLATFORM_SHARE = [0.40, 0.30, 0.30]
PLATFORM_FAIL_MULT = {1: 1.30, 2: 0.95, 3: 1.05}  # PC has the most hardware variety

AREAS = {  # area: (share of tests, base fail rate, median minutes)
    "Gameplay": (0.24, 0.060, 12),
    "Online": (0.16, 0.100, 18),
    "UI/Menus": (0.14, 0.040, 6),
    "Graphics": (0.14, 0.060, 10),
    "Audio": (0.08, 0.030, 7),
    "Career/Modes": (0.12, 0.050, 15),
    "Performance": (0.12, 0.090, 25),
}
AREA_TO_CATEGORY = {  # area -> {defect category: prob}
    "Gameplay": {"Gameplay Logic": 0.80, "Crash/Stability": 0.12, "Progression/Data": 0.08},
    "Online": {"Online/Network": 0.80, "Crash/Stability": 0.12, "Performance": 0.08},
    "UI/Menus": {"UI/UX": 0.85, "Crash/Stability": 0.08, "Progression/Data": 0.07},
    "Graphics": {"Graphics/Rendering": 0.80, "Performance": 0.12, "Crash/Stability": 0.08},
    "Audio": {"Audio": 0.88, "Crash/Stability": 0.12},
    "Career/Modes": {"Progression/Data": 0.50, "Gameplay Logic": 0.38, "Crash/Stability": 0.12},
    "Performance": {"Performance": 0.70, "Crash/Stability": 0.20, "Graphics/Rendering": 0.10},
}
SEVERITIES = ["Critical", "Major", "Minor", "Trivial"]
SEVERITY_BY_CATEGORY = {
    "Crash/Stability": [0.30, 0.40, 0.25, 0.05],
    "Online/Network": [0.12, 0.38, 0.40, 0.10],
    "Performance": [0.10, 0.40, 0.40, 0.10],
    "Progression/Data": [0.10, 0.35, 0.40, 0.15],
    "Gameplay Logic": [0.06, 0.34, 0.45, 0.15],
    "Graphics/Rendering": [0.04, 0.26, 0.50, 0.20],
    "UI/UX": [0.02, 0.20, 0.50, 0.28],
    "Audio": [0.01, 0.14, 0.50, 0.35],
}
RES_MEDIAN_DAYS = {"Critical": 1.0, "Major": 2.5, "Minor": 6.0, "Trivial": 10.0}
RES_CATEGORY_MULT = {
    "Online/Network": 1.4, "Performance": 1.3, "Crash/Stability": 1.1,
    "Graphics/Rendering": 1.1, "Progression/Data": 1.1, "Gameplay Logic": 1.0,
    "UI/UX": 0.8, "Audio": 0.8,
}
REOPEN_PROB = {
    "Online/Network": 0.18, "Performance": 0.15, "Crash/Stability": 0.12,
    "Progression/Data": 0.12, "Gameplay Logic": 0.10, "Graphics/Rendering": 0.09,
    "UI/UX": 0.06, "Audio": 0.05,
}
TEAMS = ["Functional QA", "Compatibility QA", "Regression QA", "Automation"]
CATEGORIES = list(SEVERITY_BY_CATEGORY)


def make_builds(rng: np.random.Generator) -> pd.DataFrame:
    planned = pd.Timestamp("2024-01-15") + pd.to_timedelta(14 * np.arange(N_BUILDS), "D")
    cycle_pos = np.arange(N_BUILDS) % 10
    build_type = np.where(cycle_pos < 3, "Alpha", np.where(cycle_pos < 7, "Beta", "RC"))
    change_size = np.round(rng.lognormal(mean=4.2, sigma=0.55, size=N_BUILDS)).astype(int)
    z = (np.log(change_size) - np.log(change_size).mean()) / np.log(change_size).std()
    type_effect = pd.Series(build_type).map({"Alpha": 0.20, "Beta": 0.0, "RC": -0.25}).to_numpy()
    noise = np.zeros(N_BUILDS)
    for i in range(1, N_BUILDS):  # autocorrelated "team form"
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.15)
    mult = np.exp(0.30 * z + noise + type_effect)
    for i in (22, 40):  # two regression-heavy builds for the story
        mult[i] *= 1.6
    return pd.DataFrame({
        "build_id": [f"B{i + 1:03d}" for i in range(N_BUILDS)],
        "version": [f"1.{i // 10}.{i % 10}" for i in range(N_BUILDS)],
        "build_type": build_type,
        "change_size": change_size,
        "planned_release_date": planned.date,
        "qmult": mult,
    })


def make_testers(rng: np.random.Generator, n: int = 36) -> pd.DataFrame:
    return pd.DataFrame({
        "tester_id": [f"T{i + 1:03d}" for i in range(n)],
        "team": rng.choice(TEAMS, size=n, p=[0.35, 0.25, 0.25, 0.15]),
        "experience_years": np.round(rng.uniform(0.5, 9, size=n), 1),
        "primary_platform_id": rng.choice([1, 2, 3], size=n, p=PLATFORM_SHARE),
    })


def make_test_cases(rng: np.random.Generator, n: int = 420) -> pd.DataFrame:
    areas = list(AREAS)
    area = rng.choice(areas, size=n, p=[AREAS[a][0] for a in areas])
    return pd.DataFrame({
        "test_case_id": [f"TC{i + 1:04d}" for i in range(n)],
        "area": area,
        "priority": rng.choice(["P1", "P2", "P3"], size=n, p=[0.2, 0.5, 0.3]),
    })


def make_executions(rng, builds, testers, cases) -> pd.DataFrame:
    rows = []
    tester_exp = testers.set_index("tester_id")["experience_years"]
    cases_by_area = {a: g["test_case_id"].to_numpy() for a, g in cases.groupby("area")}
    areas = list(AREAS)
    for b in builds.itertuples():
        n = rng.poisson(EXEC_PER_BUILD)
        planned = pd.Timestamp(b.planned_release_date)
        start = planned - pd.Timedelta(days=TEST_WINDOW_DAYS)
        offsets = np.sort(rng.uniform(0, TEST_WINDOW_DAYS * 24 * 3600, size=n))
        area = rng.choice(areas, size=n, p=[AREAS[a][0] for a in areas])
        platform = rng.choice([1, 2, 3], size=n, p=PLATFORM_SHARE)
        tester = rng.choice(testers["tester_id"], size=n)
        exp = tester_exp.loc[tester].to_numpy()
        med = np.array([AREAS[a][2] for a in area], dtype=float)
        dur = med * rng.lognormal(0, 0.5, size=n) * np.where(exp < 2, 1.15, 1.0)
        dur_factor = (dur / med) ** 0.25  # longer-than-usual runs find a bit more
        base = np.array([AREAS[a][1] for a in area])
        plat = np.array([PLATFORM_FAIL_MULT[p] for p in platform])
        pc_online = np.where((platform == 1) & np.isin(area, ["Online", "Graphics"]), 1.2, 1.0)
        p_fail = np.clip(0.8 * base * plat * pc_online * b.qmult * (1 + 0.03 * (exp - 3)) * dur_factor, 0, 0.6)
        u = rng.random(n)
        status = np.where(u < p_fail, "Failed", np.where(u > 0.985, "Blocked", "Passed"))
        rows.append(pd.DataFrame({
            "build_id": b.build_id,
            "test_case_id": [rng.choice(cases_by_area[a]) for a in area],
            "area": area,
            "platform_id": platform,
            "tester_id": tester,
            "status": status,
            "duration_min": np.round(dur, 1),
            "executed_at": start + pd.to_timedelta(offsets, "s"),
        }))
    ex = pd.concat(rows, ignore_index=True)
    ex.insert(0, "execution_id", [f"TX{i + 1:06d}" for i in range(len(ex))])
    return ex


def make_defects(rng, builds, executions) -> pd.DataFrame:
    mult = builds.set_index("build_id")["qmult"]
    planned = builds.set_index("build_id")["planned_release_date"].map(pd.Timestamp)
    snapshot = planned.max() + pd.Timedelta(days=7)
    rows = []

    failed = executions[executions["status"] == "Failed"]
    linked = failed[rng.random(len(failed)) < 0.65]  # rest: duplicates / script errors
    for r in linked.itertuples():
        cats, probs = zip(*AREA_TO_CATEGORY[r.area].items())
        rows.append((r.build_id, r.platform_id, r.execution_id, rng.choice(cats, p=probs),
                     r.executed_at + pd.Timedelta(hours=float(rng.uniform(0.2, 6)))))
    for b in builds.itertuples():  # exploratory finds not tied to a failed run
        for _ in range(rng.poisson(0.02 * EXEC_PER_BUILD * b.qmult)):
            start = pd.Timestamp(b.planned_release_date) - pd.Timedelta(days=TEST_WINDOW_DAYS)
            rows.append((b.build_id, rng.choice([1, 2, 3], p=PLATFORM_SHARE), None,
                         rng.choice(CATEGORIES, p=[.22, .14, .08, .14, .05, .08, .1, .19]),
                         start + pd.Timedelta(seconds=float(rng.uniform(0, TEST_WINDOW_DAYS * 86400)))))

    d = pd.DataFrame(rows, columns=["build_id", "platform_id", "execution_id", "category", "opened_at"])
    d["severity"] = [rng.choice(SEVERITIES, p=SEVERITY_BY_CATEGORY[c]) for c in d["category"]]
    sigma = 0.6
    med = d["severity"].map(RES_MEDIAN_DAYS) * d["category"].map(RES_CATEGORY_MULT)
    med = med * np.where(d["platform_id"] == 1, 1.1, 1.0)
    d["reopen_count"] = [int(rng.random() < REOPEN_PROB[c]) + int(rng.random() < REOPEN_PROB[c] * 0.25)
                         for c in d["category"]]
    res_days = rng.lognormal(np.log(med), sigma) * (1 + 0.6 * d["reopen_count"])
    d["resolved_at"] = d["opened_at"] + pd.to_timedelta(res_days, "D")
    deferred = d["severity"].isin(["Minor", "Trivial"]) & (rng.random(len(d)) < 0.12)
    d.loc[deferred, "resolved_at"] = pd.NaT
    d.loc[d["resolved_at"] > snapshot, "resolved_at"] = pd.NaT
    d["status"] = np.where(d["resolved_at"].isna(), "Open",
                           np.where(rng.random(len(d)) < 0.7, "Verified", "Fixed"))
    d["blocks_release"] = d["severity"].eq("Critical") | (
        d["severity"].eq("Major") & d["build_id"].map(builds.set_index("build_id")["build_type"]).eq("RC"))
    d = d.sort_values("opened_at", ignore_index=True)
    d.insert(0, "defect_id", [f"BUG{i + 1:05d}" for i in range(len(d))])
    return d, snapshot


def apply_release_delays(builds: pd.DataFrame, defects: pd.DataFrame, snapshot) -> pd.DataFrame:
    """Delay = how long the slowest blocking defect of a build stays open past the plan."""
    blk = defects[defects["blocks_release"] & defects["resolved_at"].notna()].copy()
    blk["planned"] = blk["build_id"].map(builds.set_index("build_id")["planned_release_date"].map(pd.Timestamp))
    blk["over"] = (blk["resolved_at"] - blk["planned"]).dt.total_seconds() / 86400
    delay = blk.groupby("build_id")["over"].max().clip(lower=0).apply(np.ceil).clip(upper=21)
    builds = builds.copy()
    builds["release_delay_days"] = builds["build_id"].map(delay).fillna(0).astype(int)
    builds["release_date"] = (pd.to_datetime(builds["planned_release_date"])
                              + pd.to_timedelta(builds["release_delay_days"], "D")).dt.date
    return builds


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    builds = make_builds(rng)
    testers = make_testers(rng)
    cases = make_test_cases(rng)
    executions = make_executions(rng, builds, testers, cases)
    defects, snapshot = make_defects(rng, builds, executions)
    builds = apply_release_delays(builds, defects, snapshot)

    executions = executions.drop(columns=["area"])  # area lives in test_cases
    builds = builds.drop(columns=["qmult"])
    PLATFORMS.to_csv(OUT / "platforms.csv", index=False)
    builds.to_csv(OUT / "builds.csv", index=False)
    testers.to_csv(OUT / "testers.csv", index=False)
    cases.to_csv(OUT / "test_cases.csv", index=False)
    executions.to_csv(OUT / "test_executions.csv", index=False)
    defects.to_csv(OUT / "defects.csv", index=False)
    (OUT / "snapshot_date.txt").write_text(str(snapshot.date()))

    done = executions[executions["status"] != "Blocked"]
    print(f"builds={len(builds)} testers={len(testers)} executions={len(executions)} defects={len(defects)}")
    print(f"overall pass rate (excl. blocked): {(done['status'] == 'Passed').mean():.3%}")
    print(f"open defects: {(defects['status'] == 'Open').sum()}  critical: {(defects['severity'] == 'Critical').sum()}")
    print(f"builds with release delay: {(builds['release_delay_days'] > 0).sum()}")


if __name__ == "__main__":
    main()
