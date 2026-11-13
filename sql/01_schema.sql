-- Schema for the SYNTHETIC game-QA dataset.
DROP TABLE IF EXISTS defects, test_executions, test_cases, testers, builds, platforms CASCADE;

CREATE TABLE platforms (
    platform_id   SMALLINT PRIMARY KEY,
    platform_name TEXT NOT NULL UNIQUE
);

CREATE TABLE builds (
    build_id             TEXT PRIMARY KEY,
    version              TEXT NOT NULL,
    build_type           TEXT NOT NULL CHECK (build_type IN ('Alpha','Beta','RC')),
    change_size          INTEGER NOT NULL CHECK (change_size > 0),  -- changed work items in the build
    planned_release_date DATE NOT NULL,
    release_delay_days   INTEGER NOT NULL DEFAULT 0 CHECK (release_delay_days >= 0),
    release_date         DATE NOT NULL
);

CREATE TABLE testers (
    tester_id           TEXT PRIMARY KEY,
    team                TEXT NOT NULL,
    experience_years    NUMERIC(3,1) NOT NULL,
    primary_platform_id SMALLINT REFERENCES platforms
);

CREATE TABLE test_cases (
    test_case_id TEXT PRIMARY KEY,
    area         TEXT NOT NULL,
    priority     TEXT NOT NULL CHECK (priority IN ('P1','P2','P3'))
);

CREATE TABLE test_executions (
    execution_id TEXT PRIMARY KEY,
    build_id     TEXT NOT NULL REFERENCES builds,
    test_case_id TEXT NOT NULL REFERENCES test_cases,
    platform_id  SMALLINT NOT NULL REFERENCES platforms,
    tester_id    TEXT NOT NULL REFERENCES testers,
    status       TEXT NOT NULL CHECK (status IN ('Passed','Failed','Blocked')),
    duration_min NUMERIC(7,1) NOT NULL CHECK (duration_min > 0),
    executed_at  TIMESTAMP NOT NULL
);

CREATE TABLE defects (
    defect_id      TEXT PRIMARY KEY,
    build_id       TEXT NOT NULL REFERENCES builds,
    platform_id    SMALLINT NOT NULL REFERENCES platforms,
    execution_id   TEXT REFERENCES test_executions,   -- NULL = exploratory find
    category       TEXT NOT NULL,
    severity       TEXT NOT NULL CHECK (severity IN ('Critical','Major','Minor','Trivial')),
    opened_at      TIMESTAMP NOT NULL,
    resolved_at    TIMESTAMP,                          -- NULL = still open
    reopen_count   SMALLINT NOT NULL DEFAULT 0,
    status         TEXT NOT NULL CHECK (status IN ('Open','Fixed','Verified')),
    blocks_release BOOLEAN NOT NULL,
    CHECK ((status = 'Open') = (resolved_at IS NULL)),
    CHECK (resolved_at IS NULL OR resolved_at >= opened_at)
);

CREATE INDEX ix_exec_build    ON test_executions (build_id);
CREATE INDEX ix_exec_platform ON test_executions (platform_id);
CREATE INDEX ix_def_build     ON defects (build_id);
CREATE INDEX ix_def_platform  ON defects (platform_id);
CREATE INDEX ix_def_category  ON defects (category);
