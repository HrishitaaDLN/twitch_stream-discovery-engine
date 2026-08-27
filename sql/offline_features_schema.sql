-- Offline feature tables, populated by scripts/compute_offline_features.py.
--
-- These are batch snapshots -- recomputed periodically (nightly, in a real
-- deployment) from Phase 1 historical data, not updated per-event. They live
-- in Postgres rather than Redis for three reasons:
--   1. A TTL is the wrong semantic here: a batch estimate is valid until the
--      NEXT batch run, not for a fixed decay window. Letting it expire would
--      just replace a good (if slightly stale) estimate with a cold-start
--      default, which is strictly worse.
--   2. The source data (watch_events, follow_events, streams) already lives
--      in Postgres, so computing here avoids an extra ETL hop.
--   3. Row counts are small (tens of thousands at most) -- Postgres serves
--      point lookups fast, and the same tables are directly queryable for
--      analytics/dashboards, not just feature serving.

DROP TABLE IF EXISTS viewer_streamer_affinity CASCADE;
DROP TABLE IF EXISTS streamer_features_offline CASCADE;
DROP TABLE IF EXISTS category_popularity_offline CASCADE;
DROP TABLE IF EXISTS feature_store_globals CASCADE;

-- Grain: (viewer, streamer). Source for the "streamer_affinity" feature --
-- how much this specific viewer has historically watched this specific streamer.
CREATE TABLE viewer_streamer_affinity (
    viewer_id            INTEGER NOT NULL,
    streamer_id           INTEGER NOT NULL,
    total_watch_seconds     BIGINT NOT NULL,
    avg_watch_seconds        REAL NOT NULL,
    session_count             INTEGER NOT NULL,
    last_watched_at            TIMESTAMP NOT NULL,
    computed_at                 TIMESTAMP NOT NULL,
    PRIMARY KEY (viewer_id, streamer_id)
);
CREATE INDEX idx_vsa_streamer ON viewer_streamer_affinity (streamer_id);

-- Grain: streamer. streamer_growth_7d and a follow-conversion prior
-- (a heuristic stand-in for a future dedicated follow-probability model).
CREATE TABLE streamer_features_offline (
    streamer_id                  INTEGER PRIMARY KEY,
    growth_7d                      REAL NOT NULL,
    follow_conversion_rate          REAL NOT NULL,
    historical_viewer_count          INTEGER NOT NULL,
    historical_follower_count         INTEGER NOT NULL,
    computed_at                        TIMESTAMP NOT NULL
);

-- Grain: category. Empirical (not synthetic-generator) popularity, used as
-- the cold-start-viewer fallback for viewer_game_affinity.
CREATE TABLE category_popularity_offline (
    category_id      SMALLINT PRIMARY KEY,
    popularity_share    REAL NOT NULL,
    computed_at           TIMESTAMP NOT NULL
);

-- Single-row table of dataset-wide priors, used when even the streamer-level
-- table has no entry (a genuinely unseen streamer_id).
CREATE TABLE feature_store_globals (
    id                                SMALLINT PRIMARY KEY DEFAULT 1,
    global_avg_watch_seconds            REAL NOT NULL,
    global_follow_conversion_rate         REAL NOT NULL,
    computed_at                            TIMESTAMP NOT NULL,
    CHECK (id = 1)
);
