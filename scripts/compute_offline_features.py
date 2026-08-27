"""
Phase 3 offline batch job.

Reads Phase 1 historical Postgres tables (watch_events, streams, follow_events)
and computes the offline/batch feature tables defined in
sql/offline_features_schema.sql:

  viewer_streamer_affinity     (viewer, streamer) -> historical watch behavior
  streamer_features_offline    streamer -> 7d growth, follow-conversion prior
  category_popularity_offline  category -> empirical share of watch volume
  feature_store_globals        single-row dataset-wide priors

Idempotent: TRUNCATEs each table before writing, so re-running (as a nightly
job would) always reflects the current state of the source tables.

Run: python scripts/compute_offline_features.py
"""
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://streammatch:streammatch@localhost:5433/streammatch"
)

MIN_EVENTS_FOR_GROWTH = 10   # below this, a growth estimate is just noise
GROWTH_CLIP = (-2.0, 5.0)    # -200% to +500%, keeps outliers from dominating the model


def main():
    conn = psycopg2.connect(DATABASE_URL)
    now = datetime.now(timezone.utc)

    print("Reading watch_events joined with streams...")
    watch = pd.read_sql("""
        SELECT w.viewer_id, w.watch_duration_seconds, w.session_start,
               s.streamer_id, s.category_id
        FROM watch_events w
        JOIN streams s ON w.stream_id = s.stream_id
    """, conn)

    # ---- 1. viewer_streamer_affinity ------------------------------------
    print("Computing viewer_streamer_affinity...")
    vsa = (
        watch.groupby(["viewer_id", "streamer_id"])
        .agg(
            total_watch_seconds=("watch_duration_seconds", "sum"),
            avg_watch_seconds=("watch_duration_seconds", "mean"),
            session_count=("watch_duration_seconds", "count"),
            last_watched_at=("session_start", "max"),
        )
        .reset_index()
    )
    print(f"  {len(vsa):,} (viewer, streamer) pairs")

    # ---- 2. streamer_features_offline: growth_7d -------------------------
    print("Computing streamer_growth_7d...")
    watch["day"] = pd.to_datetime(watch["session_start"]).dt.date
    daily = watch.groupby(["streamer_id", "day"]).size().reset_index(name="n")
    max_day = daily["day"].max()
    recent_start = max_day - pd.Timedelta(days=6)
    prev_start = max_day - pd.Timedelta(days=13)
    prev_end = max_day - pd.Timedelta(days=7)

    growth_rows = []
    for streamer_id, g in daily.groupby("streamer_id"):
        total_events = g["n"].sum()
        recent = g.loc[g["day"] >= recent_start, "n"].sum()
        prev = g.loc[(g["day"] >= prev_start) & (g["day"] <= prev_end), "n"].sum()
        if total_events < MIN_EVENTS_FOR_GROWTH or prev == 0:
            growth = 0.0  # insufficient history -> neutral, not a noisy extreme
        else:
            growth = float(np.clip((recent / 7.0 - prev / 7.0) / (prev / 7.0), *GROWTH_CLIP))
        growth_rows.append({"streamer_id": streamer_id, "growth_7d": growth})
    growth_df = pd.DataFrame(growth_rows)

    # ---- 3. streamer_features_offline: follow_conversion_rate ------------
    print("Computing follow_conversion_rate...")
    viewers_per_streamer = watch.groupby("streamer_id")["viewer_id"].nunique().reset_index(name="historical_viewer_count")
    follow = pd.read_sql("SELECT streamer_id, viewer_id FROM follow_events", conn)
    followers_per_streamer = follow.groupby("streamer_id")["viewer_id"].nunique().reset_index(name="historical_follower_count")

    streamer_offline = viewers_per_streamer.merge(followers_per_streamer, on="streamer_id", how="left")
    streamer_offline["historical_follower_count"] = streamer_offline["historical_follower_count"].fillna(0).astype(int)
    streamer_offline["follow_conversion_rate"] = (
        streamer_offline["historical_follower_count"] / streamer_offline["historical_viewer_count"]
    )
    streamer_offline = streamer_offline.merge(growth_df, on="streamer_id", how="left")
    streamer_offline["growth_7d"] = streamer_offline["growth_7d"].fillna(0.0)

    # ---- 4. dataset-wide globals -------------------------------------------
    print("Computing global priors...")
    global_avg_watch = float(watch["watch_duration_seconds"].mean())
    total_pairs = len(vsa)
    total_follow_pairs = len(follow.drop_duplicates(["viewer_id", "streamer_id"]))
    global_follow_rate = total_follow_pairs / total_pairs if total_pairs else 0.05
    streamer_offline["follow_conversion_rate"] = streamer_offline["follow_conversion_rate"].fillna(global_follow_rate)

    # ---- 5. category popularity (empirical, not the synthetic generator's) --
    print("Computing category_popularity_offline...")
    cat_pop = watch.groupby("category_id").size().reset_index(name="n")
    cat_pop["popularity_share"] = cat_pop["n"] / cat_pop["n"].sum()

    # ---- write everything ---------------------------------------------------
    with conn.cursor() as cur:
        print("Writing viewer_streamer_affinity...")
        cur.execute("TRUNCATE viewer_streamer_affinity")
        execute_values(
            cur,
            """INSERT INTO viewer_streamer_affinity
               (viewer_id, streamer_id, total_watch_seconds, avg_watch_seconds,
                session_count, last_watched_at, computed_at)
               VALUES %s""",
            [
                (int(r.viewer_id), int(r.streamer_id), int(r.total_watch_seconds),
                 float(r.avg_watch_seconds), int(r.session_count), r.last_watched_at, now)
                for r in vsa.itertuples()
            ],
        )

        print("Writing streamer_features_offline...")
        cur.execute("TRUNCATE streamer_features_offline")
        execute_values(
            cur,
            """INSERT INTO streamer_features_offline
               (streamer_id, growth_7d, follow_conversion_rate,
                historical_viewer_count, historical_follower_count, computed_at)
               VALUES %s""",
            [
                (int(r.streamer_id), float(r.growth_7d), float(r.follow_conversion_rate),
                 int(r.historical_viewer_count), int(r.historical_follower_count), now)
                for r in streamer_offline.itertuples()
            ],
        )

        print("Writing category_popularity_offline...")
        cur.execute("TRUNCATE category_popularity_offline")
        execute_values(
            cur,
            "INSERT INTO category_popularity_offline (category_id, popularity_share, computed_at) VALUES %s",
            [(int(r.category_id), float(r.popularity_share), now) for r in cat_pop.itertuples()],
        )

        print("Writing feature_store_globals...")
        cur.execute("TRUNCATE feature_store_globals")
        cur.execute(
            """INSERT INTO feature_store_globals
               (id, global_avg_watch_seconds, global_follow_conversion_rate, computed_at)
               VALUES (1, %s, %s, %s)""",
            (global_avg_watch, global_follow_rate, now),
        )

    conn.commit()
    conn.close()

    print("\nDone.")
    print(f"  viewer_streamer_affinity: {len(vsa):,} rows")
    print(f"  streamer_features_offline: {len(streamer_offline):,} rows")
    print(f"  category_popularity_offline: {len(cat_pop):,} rows")
    print(f"  global_avg_watch_seconds: {global_avg_watch:.1f}")
    print(f"  global_follow_conversion_rate: {global_follow_rate:.4f}")


if __name__ == "__main__":
    main()
