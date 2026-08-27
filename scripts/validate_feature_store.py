"""
Phase 3 validation: call FeatureStore.get_features for a handful of
(viewer, stream) pairs -- warm cases and explicit cold-start cases -- and
print the full feature vector so it can be eyeballed.

Run: python scripts/validate_feature_store.py
"""
import json
import psycopg2

from feature_store import FeatureStore, DATABASE_URL


def pick_warm_historical_pair(conn):
    """A (viewer, stream) pair with real Phase 1 watch history behind it."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT viewer_id, stream_id FROM watch_events
            WHERE watch_duration_seconds > 600
            ORDER BY event_id LIMIT 1 OFFSET 5000
        """)
        return cur.fetchone()


def pick_repeat_viewer_pair(conn):
    """A (viewer, streamer)->stream pair where the viewer has watched this
    streamer many times, so streamer_affinity should be well-populated."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT viewer_id, streamer_id, session_count FROM viewer_streamer_affinity
            ORDER BY session_count DESC LIMIT 1
        """)
        viewer_id, streamer_id, session_count = cur.fetchone()
        cur.execute("SELECT stream_id FROM streams WHERE streamer_id = %s LIMIT 1", (streamer_id,))
        stream_id = cur.fetchone()[0]
        return viewer_id, stream_id, session_count


def pick_live_stream():
    """A stream_id currently live from the Phase 2 producer (Redis-only, not
    yet in Postgres `streams`) -- exercises the redis-fallback stream context
    path, and should show WARM online features (recent ticks/chat) if the
    producer/consumer were run recently, since Redis state expires."""
    fs_probe = FeatureStore()
    ids = fs_probe.redis.zrevrange("live_streams_by_viewers", 0, 0)
    fs_probe.close()
    return int(ids[0]) if ids else None


def main():
    conn = psycopg2.connect(DATABASE_URL)
    fs = FeatureStore()

    cases = []

    warm_pair = pick_warm_historical_pair(conn)
    cases.append(("WARM: real historical (viewer, stream) pair, long watch on record",
                   warm_pair[0], warm_pair[1]))

    repeat_viewer, repeat_stream, session_count = pick_repeat_viewer_pair(conn)
    cases.append((f"WARM: viewer with {session_count} historical sessions on this streamer "
                   f"(streamer_affinity should reflect real history, not the global default)",
                   repeat_viewer, repeat_stream))

    live_stream_id = pick_live_stream()
    if live_stream_id:
        cases.append((f"LIVE: Phase 2 producer-only stream (stream_id={live_stream_id}, "
                       f"not in Postgres `streams` -- exercises the Redis stream-context fallback)",
                       repeat_viewer, live_stream_id))
    else:
        print("(No live Phase 2 stream currently in Redis -- run producer.py briefly first "
              "to see the Redis-fallback stream-context case.)\n")

    cases.append(("COLD START: brand-new viewer_id (no signup ever went through the system), "
                   "real historical stream",
                   999001, warm_pair[1]))

    cases.append(("COLD START: real viewer, completely unknown stream_id "
                   "(not in Postgres, not in Redis -- e.g. stream just created, no ticks yet)",
                   warm_pair[0], 88888888))

    cases.append(("COLD START: both viewer AND stream unknown (worst case)",
                   999002, 88888889))

    for label, viewer_id, stream_id in cases:
        print("=" * 100)
        print(f"{label}")
        print(f"  get_features(viewer_id={viewer_id}, stream_id={stream_id})")
        print("-" * 100)
        features = fs.get_features(viewer_id, stream_id)
        print(json.dumps(features, indent=2, default=str))
        print()

    fs.close()
    conn.close()


if __name__ == "__main__":
    main()
