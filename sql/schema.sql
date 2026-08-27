-- StreamMatch Phase 1 schema
-- Design principle: these tables hold raw ground-truth entities and interaction
-- events only. Derived features (viewer_game_affinity, rolling watch_duration,
-- streamer_growth, chat_velocity, etc.) are computed FROM this data in the
-- feature pipeline (Phase 2/3), not stored here.

DROP TABLE IF EXISTS chat_events CASCADE;
DROP TABLE IF EXISTS follow_events CASCADE;
DROP TABLE IF EXISTS watch_events CASCADE;
DROP TABLE IF EXISTS streams CASCADE;
DROP TABLE IF EXISTS viewer_category_affinity CASCADE;
DROP TABLE IF EXISTS viewers CASCADE;
DROP TABLE IF EXISTS streamers CASCADE;
DROP TABLE IF EXISTS categories CASCADE;

CREATE TABLE categories (
    category_id     SMALLINT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    is_irl          BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE streamers (
    streamer_id             INTEGER PRIMARY KEY,
    username                TEXT NOT NULL UNIQUE,
    primary_category_id     SMALLINT NOT NULL REFERENCES categories(category_id),
    secondary_category_id   SMALLINT REFERENCES categories(category_id),
    language                TEXT NOT NULL,
    popularity_tier         TEXT NOT NULL CHECK (popularity_tier IN ('micro','small','mid','large','mega')),
    account_created_at      TIMESTAMP NOT NULL
);

CREATE TABLE viewers (
    viewer_id           INTEGER PRIMARY KEY,
    username             TEXT NOT NULL UNIQUE,
    primary_language     TEXT NOT NULL,
    viewer_type          TEXT NOT NULL CHECK (viewer_type IN ('casual','regular','power')),
    signup_at            TIMESTAMP NOT NULL
);

-- Generative ground truth for viewer preference, NOT a served feature.
-- Sums to ~1.0 across a viewer's rows (Dirichlet-sampled).
CREATE TABLE viewer_category_affinity (
    viewer_id       INTEGER NOT NULL REFERENCES viewers(viewer_id),
    category_id     SMALLINT NOT NULL REFERENCES categories(category_id),
    affinity_score  REAL NOT NULL,
    PRIMARY KEY (viewer_id, category_id)
);

CREATE TABLE streams (
    stream_id       BIGINT PRIMARY KEY,
    streamer_id     INTEGER NOT NULL REFERENCES streamers(streamer_id),
    category_id     SMALLINT NOT NULL REFERENCES categories(category_id),
    title           TEXT NOT NULL,
    language        TEXT NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    ended_at        TIMESTAMP  -- NULL = still live at snapshot time
);

CREATE TABLE watch_events (
    event_id                BIGINT PRIMARY KEY,
    viewer_id                INTEGER NOT NULL REFERENCES viewers(viewer_id),
    stream_id                BIGINT NOT NULL REFERENCES streams(stream_id),
    session_start             TIMESTAMP NOT NULL,
    watch_duration_seconds    INTEGER NOT NULL,
    label_watched_gt_5min     BOOLEAN NOT NULL
);

CREATE TABLE chat_events (
    event_id       BIGINT PRIMARY KEY,
    viewer_id       INTEGER NOT NULL REFERENCES viewers(viewer_id),
    stream_id       BIGINT NOT NULL REFERENCES streams(stream_id),
    message_text     TEXT NOT NULL,
    ts               TIMESTAMP NOT NULL
);

CREATE TABLE follow_events (
    event_id       BIGINT PRIMARY KEY,
    viewer_id       INTEGER NOT NULL REFERENCES viewers(viewer_id),
    streamer_id     INTEGER NOT NULL REFERENCES streamers(streamer_id),
    followed_at      TIMESTAMP NOT NULL,
    UNIQUE (viewer_id, streamer_id)
);

-- Indexes for the access patterns Phase 3+ will actually use:
-- feature retrieval by viewer, by stream, and time-windowed rollups.
CREATE INDEX idx_watch_events_viewer ON watch_events (viewer_id, session_start);
CREATE INDEX idx_watch_events_stream ON watch_events (stream_id, session_start);
CREATE INDEX idx_chat_events_stream ON chat_events (stream_id, ts);
CREATE INDEX idx_chat_events_viewer ON chat_events (viewer_id, ts);
CREATE INDEX idx_follow_events_viewer ON follow_events (viewer_id);
CREATE INDEX idx_follow_events_streamer ON follow_events (streamer_id);
CREATE INDEX idx_streams_streamer ON streams (streamer_id, started_at);
CREATE INDEX idx_streams_category ON streams (category_id, started_at);
CREATE INDEX idx_streams_live ON streams (started_at, ended_at);
