"""
Phase 3: the single feature retrieval layer.

FeatureStore.get_features(viewer_id, stream_id) is the ONE code path both
Phase 5 (offline model training, called in a loop over historical
(viewer,stream) pairs) and Phase 6 (the live inference API) will use to
assemble a feature vector. There is no separate "training feature logic" --
whatever this method returns is exactly what the model sees, always.

Design:
- Online features (concurrent_viewers, chat_velocity, recent_categories,
  watch_duration_ema) are read from Redis, written by the Phase 2 consumer.
- Offline features (viewer_game_affinity, streamer_affinity, streamer_growth,
  follow_probability prior, category popularity) are read from small
  in-memory caches loaded once from the Postgres tables Phase 3's batch job
  (compute_offline_features.py) populates. Caching avoids N+1 Postgres round
  trips per inference call -- call `.refresh()` after each batch run to pick
  up new numbers, mirroring how a real serving layer reloads a materialized
  snapshot on a schedule rather than hitting the warehouse per request.
- time_of_day is computed at request time, never stored anywhere.

COLD START -- the actual design decisions, not just "handle it":

  New/unknown VIEWER (no row in viewer_category_affinity, e.g. a signup that
  hasn't been through a batch refresh yet):
    - viewer_game_affinity  -> falls back to category_popularity_offline
      (the empirical population-wide share for that category), NOT 0.
      Reasoning: 0 says "this viewer actively dislikes every category,"
      which would suppress recommendations for every new user. A population
      prior says "we don't know yet, assume average," which is what you'd
      actually want a cold-start user to see (popular, broadly-appealing
      content) until real signal arrives.

  New/unknown (VIEWER, STREAMER) pair (viewer exists, but has never watched
  this particular streamer):
    - streamer_affinity     -> falls back to global_avg_watch_seconds,
      NOT 0. Reasoning: 0 seconds reads as "this viewer bounces immediately
      from this streamer," which is a strong negative signal the model would
      learn to penalize -- but we have no evidence of that, only an absence
      of data. The global average says "assume typical engagement" instead
      of "assume the worst."
    - streamer_affinity_sessions stays a real 0 -- "zero sessions" IS the
      true, correct value here, unlike watch duration there's no misleading
      interpretation risk.
    - follow_probability    -> falls back to the streamer's own historical
      follow_conversion_rate (or the dataset-wide rate if the streamer is
      also unseen). This is an explicit placeholder for a future dedicated
      P(follow | viewer, streamer) model; documented as such below.

  New/unknown STREAM (just went live, Phase 2 consumer hasn't ticked yet, or
  a stream_id this system has never seen at all):
    - concurrent_viewers    -> falls back to DEFAULT_CONCURRENT_VIEWERS (a
      small nonzero constant), NOT 0. Reasoning: a literal 0 creates a
      cold-start trap -- ranking/candidate-gen would treat a brand new
      stream as "nobody's watching, rank it last," which means it can never
      accumulate the very viewers who'd prove it's worth watching (a classic
      recsys rich-get-richer failure). A small nonzero floor says "plausibly
      being discovered" without falsely inflating it above real hot streams.
    - chat_velocity         -> 0.0 IS the correct default here. Unlike watch
      duration, an empty chat window in the last N minutes is a genuinely
      accurate observation, not a misleading stand-in for "no data."
    - streamer_growth       -> falls back to 0.0 (neutral: "no evidence of
      growth or decline") rather than an extreme, noise-driven value.
    - category_id / streamer_id / language for the stream itself: resolved
      from Postgres `streams` (historical/batch-known streams) first, then
      Redis `stream:{id}:meta` (Phase 2 producer's live-only streams that
      haven't been through a batch sync yet -- a deliberate illustration of
      batch/streaming staleness). If neither has it, they're None and every
      downstream feature that depends on them uses its own fallback.
"""
import os
import math
from datetime import datetime, timezone

import psycopg2
import redis


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://streammatch:streammatch@localhost:5433/streammatch"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

DEFAULT_CONCURRENT_VIEWERS = 3   # small nonzero floor -- see module docstring
RECENT_CATEGORIES_N = 10
CHAT_VELOCITY_WINDOW_SEC = 120.0  # must match consumer.py's --chat-window-sec for consistency


class FeatureStore:
    def __init__(self, database_url=DATABASE_URL, redis_url=REDIS_URL):
        self.pg = psycopg2.connect(database_url)
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.refresh()

    def refresh(self):
        """Reload all in-memory offline caches from Postgres. Call this after
        each run of compute_offline_features.py to pick up fresh numbers."""
        with self.pg.cursor() as cur:
            cur.execute("SELECT viewer_id, primary_language FROM viewers")
            self.viewer_language = dict(cur.fetchall())

            cur.execute("SELECT viewer_id, category_id, affinity_score FROM viewer_category_affinity")
            self.viewer_category_affinity = {(v, c): s for v, c, s in cur.fetchall()}

            cur.execute("SELECT stream_id, streamer_id, category_id, language FROM streams")
            self.stream_dim = {
                sid: {"streamer_id": st, "category_id": cat, "language": lang}
                for sid, st, cat, lang in cur.fetchall()
            }

            cur.execute("SELECT category_id, popularity_share FROM category_popularity_offline")
            self.category_popularity = dict(cur.fetchall())

            cur.execute("""SELECT viewer_id, streamer_id, avg_watch_seconds, session_count
                            FROM viewer_streamer_affinity""")
            self.viewer_streamer_affinity = {
                (v, s): {"avg_watch_seconds": avg, "session_count": cnt}
                for v, s, avg, cnt in cur.fetchall()
            }

            cur.execute("""SELECT streamer_id, growth_7d, follow_conversion_rate
                            FROM streamer_features_offline""")
            self.streamer_offline = {
                sid: {"growth_7d": g, "follow_conversion_rate": f}
                for sid, g, f in cur.fetchall()
            }

            cur.execute("""SELECT global_avg_watch_seconds, global_follow_conversion_rate
                            FROM feature_store_globals WHERE id = 1""")
            self.global_avg_watch_seconds, self.global_follow_conversion_rate = cur.fetchone()

        print(f"[FeatureStore] refreshed: {len(self.viewer_language)} viewers, "
              f"{len(self.viewer_category_affinity)} affinity rows, "
              f"{len(self.stream_dim)} known streams, "
              f"{len(self.viewer_streamer_affinity)} viewer-streamer pairs, "
              f"{len(self.streamer_offline)} streamers with offline features")

    # -- stream/viewer context resolution -----------------------------------

    def _resolve_stream_context(self, stream_id):
        row = self.stream_dim.get(stream_id)
        if row is not None:
            return {**row, "source": "postgres"}

        meta = self.redis.hgetall(f"stream:{stream_id}:meta")
        if meta:
            return {
                "streamer_id": int(meta["streamer_id"]),
                "category_id": int(meta["category_id"]),
                "language": meta["language"],
                "source": "redis",
            }

        return {"streamer_id": None, "category_id": None, "language": None, "source": "unknown"}

    # -- online features (Redis) --------------------------------------------

    def _get_online_features(self, viewer_id, stream_id):
        concurrent_raw = self.redis.get(f"stream:{stream_id}:concurrent_viewers")
        is_cold_start_stream = concurrent_raw is None
        concurrent_viewers = int(concurrent_raw) if concurrent_raw is not None else DEFAULT_CONCURRENT_VIEWERS

        chat_key = f"stream:{stream_id}:chat_ts"
        now = datetime.now(timezone.utc).timestamp()
        self.redis.zremrangebyscore(chat_key, 0, now - CHAT_VELOCITY_WINDOW_SEC)
        chat_count = self.redis.zcard(chat_key)
        chat_velocity = chat_count / (CHAT_VELOCITY_WINDOW_SEC / 60.0)

        recent_categories = [
            int(c) for c in self.redis.lrange(f"viewer:{viewer_id}:recent_categories", 0, RECENT_CATEGORIES_N - 1)
        ]

        stats = self.redis.hgetall(f"viewer:{viewer_id}:watch_stats")
        watch_duration_ema = float(stats["duration_ema"]) if stats else self.global_avg_watch_seconds

        return {
            "concurrent_viewers": concurrent_viewers,
            "chat_velocity": chat_velocity,
            "recent_categories": recent_categories,
            "watch_duration_ema": watch_duration_ema,
            "is_cold_start_stream": is_cold_start_stream,
        }

    # -- offline features (Postgres, cached in memory) -----------------------

    def _get_offline_features(self, viewer_id, streamer_id, category_id):
        is_cold_start_viewer = viewer_id not in self.viewer_language

        if category_id is not None:
            viewer_game_affinity = self.viewer_category_affinity.get(
                (viewer_id, category_id),
                self.category_popularity.get(category_id, 1.0 / max(len(self.category_popularity), 1)),
            )
        else:
            viewer_game_affinity = 1.0 / max(len(self.category_popularity), 1)  # fully unknown category

        pair = self.viewer_streamer_affinity.get((viewer_id, streamer_id)) if streamer_id is not None else None
        if pair is not None:
            streamer_affinity = pair["avg_watch_seconds"]
            streamer_affinity_sessions = pair["session_count"]
        else:
            streamer_affinity = self.global_avg_watch_seconds
            streamer_affinity_sessions = 0

        streamer_row = self.streamer_offline.get(streamer_id) if streamer_id is not None else None
        streamer_growth = streamer_row["growth_7d"] if streamer_row else 0.0

        return {
            "viewer_game_affinity": viewer_game_affinity,
            "streamer_affinity": streamer_affinity,
            "streamer_affinity_sessions": streamer_affinity_sessions,
            "streamer_growth": streamer_growth,
            "is_cold_start_viewer": is_cold_start_viewer,
        }

    def _get_follow_probability(self, viewer_id, streamer_id):
        if streamer_id is not None and self.redis.sismember(f"viewer:{viewer_id}:follows", streamer_id):
            return 1.0  # already following -- ground truth beats any prior
        streamer_row = self.streamer_offline.get(streamer_id) if streamer_id is not None else None
        if streamer_row is not None:
            return streamer_row["follow_conversion_rate"]
        return self.global_follow_conversion_rate

    # -- time features (computed, never stored) ------------------------------

    @staticmethod
    def _get_time_features():
        now = datetime.now(timezone.utc)
        hour_frac = now.hour + now.minute / 60.0
        angle = 2 * math.pi * hour_frac / 24.0
        return {"time_of_day_sin": math.sin(angle), "time_of_day_cos": math.cos(angle)}

    # -- the one public method ------------------------------------------------

    def get_features(self, viewer_id: int, stream_id: int) -> dict:
        ctx = self._resolve_stream_context(stream_id)
        streamer_id, category_id, stream_language = ctx["streamer_id"], ctx["category_id"], ctx["language"]

        online = self._get_online_features(viewer_id, stream_id)
        offline = self._get_offline_features(viewer_id, streamer_id, category_id)
        follow_probability = self._get_follow_probability(viewer_id, streamer_id)
        time_features = self._get_time_features()

        viewer_language = self.viewer_language.get(viewer_id)
        language_match = int(viewer_language is not None and viewer_language == stream_language)
        recent_category_match = int(category_id is not None and category_id in online["recent_categories"])

        features = {
            "viewer_id": viewer_id,
            "stream_id": stream_id,
            "streamer_id": streamer_id,
            "category_id": category_id,
            "viewer_language": viewer_language,
            "stream_language": stream_language,
            "language_match": language_match,
            "viewer_game_affinity": offline["viewer_game_affinity"],
            "streamer_affinity": offline["streamer_affinity"],
            "streamer_affinity_sessions": offline["streamer_affinity_sessions"],
            "streamer_growth": offline["streamer_growth"],
            "follow_probability": follow_probability,
            "concurrent_viewers": online["concurrent_viewers"],
            "chat_velocity": online["chat_velocity"],
            "recent_categories": online["recent_categories"],
            "recent_category_match": recent_category_match,
            "watch_duration_ema": online["watch_duration_ema"],
            "time_of_day_sin": time_features["time_of_day_sin"],
            "time_of_day_cos": time_features["time_of_day_cos"],
            "meta": {
                "is_cold_start_viewer": offline["is_cold_start_viewer"],
                "is_cold_start_stream": online["is_cold_start_stream"],
                "stream_source": ctx["source"],
                "computed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        return features

    def close(self):
        self.pg.close()
