"""Data-quality checks against the loaded database (needs docker compose up + load_db.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from db import engine, query_df  # noqa: E402


@pytest.fixture(scope="module")
def eng():
    return engine()


def scalar(eng, q):
    return query_df(q, eng).iloc[0, 0]


def test_no_orphan_defects(eng):
    assert scalar(eng, "SELECT COUNT(*) FROM defects d LEFT JOIN builds b USING (build_id) WHERE b.build_id IS NULL") == 0


def test_defects_not_resolved_before_opened(eng):
    assert scalar(eng, "SELECT COUNT(*) FROM defects WHERE resolved_at < opened_at") == 0


def test_open_status_matches_missing_resolution(eng):
    assert scalar(eng, "SELECT COUNT(*) FROM defects WHERE (status = 'Open') <> (resolved_at IS NULL)") == 0


def test_pass_rate_bounds(eng):
    assert scalar(eng, "SELECT COUNT(*) FROM v_build_metrics WHERE pass_rate NOT BETWEEN 0 AND 1") == 0


def test_view_totals_match_raw_tables(eng):
    assert scalar(eng, "SELECT SUM(executions) FROM v_build_metrics") == scalar(eng, "SELECT COUNT(*) FROM test_executions")
    assert scalar(eng, "SELECT SUM(defects) FROM v_build_metrics") == scalar(eng, "SELECT COUNT(*) FROM defects")
    assert scalar(eng, "SELECT SUM(defects) FROM v_platform_metrics") == scalar(eng, "SELECT COUNT(*) FROM defects")


def test_release_date_is_plan_plus_delay(eng):
    assert scalar(eng, "SELECT COUNT(*) FROM builds WHERE release_date <> planned_release_date + release_delay_days") == 0


def test_threshold_flag_consistent(eng):
    bad = scalar(eng, """SELECT COUNT(*) FROM v_build_metrics m, v_thresholds t
                         WHERE m.meets_release_quality <> (m.pass_rate >= t.min_pass_rate AND m.critical_defects <= t.max_critical)""")
    assert bad == 0
