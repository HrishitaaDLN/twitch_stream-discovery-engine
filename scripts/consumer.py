"""
StreamMatch feature-pipeline consumer (Phase 2).

Reads all four topics (watch_events, chat_events, follow_events,
stream_status) and maintains ONLINE features in Redis, chosen per access
pattern rather than as JSON blobs:

  concurrent_viewers   stream:{id}:concurrent_viewers   STRING, EX 30s
                        + live_streams_by_viewers         ZSET (score=viewers) for top-N queries
  chat_velocity        stream:{id}:chat_ts               ZSET (member=event_id, score=ts),
                        pruned to a rolling window on every read/write, self-expiring
  recent_categories    viewer:{id}:recent_categories      LIST, LPUSH + LTRIM capped at N
  watch_duration avg   viewer:{id}:watch_stats            HASH {duration_ema, session_count},
                        updated incrementally (EMA), never recomputed from scratch
  follows               viewer:{id}:follows                SET

Idempotency: every event that has a side effect gated behind
`SETNX dedup:{topic}:{event_id}` (an atomic "have I seen this before"
check with a TTL). If a message is redelivered (consumer restart,
rebalance), the dedup key already exists and the handler is a no-op --
Redis state is never double-counted. `viewer_count_tick` writes are
naturally idempotent (plain overwrites) and skip the dedup check.

Run: python scripts/consumer.py --chat-window-sec 120 --snapshot-interval-sec 30
"""
import argparse
import json
import time

import redis
from kafka import KafkaConsumer, TopicPartition

import sim_common as sc

RECENT_CATEGORIES_CAP = 10
EMA_ALPHA = 0.3
CONCURRENT_TTL_SEC = 30


class FeatureConsumer:
    def __init__(self, r: redis.Redis, chat_window_sec: float):
        self.r = r
        self.chat_window_sec = chat_window_sec
        self.counts = {"watch": 0, "chat": 0, "follow": 0, "stream_started": 0,
                       "stream_ended": 0, "tick": 0, "dupes_skipped": 0}

    def _is_new(self, topic, event_id, ttl=3600):
        key = f"dedup:{topic}:{event_id}"
        is_new = self.r.set(key, "1", nx=True, ex=ttl)
        if not is_new:
            self.counts["dupes_skipped"] += 1
        return bool(is_new)

    def handle_watch_event(self, msg):
        if not self._is_new("watch_events", msg["event_id"]):
            return
        viewer_id = msg["viewer_id"]
        category_id = msg["category_id"]
        duration = msg["watch_duration_seconds"]

        pipe = self.r.pipeline()
        pipe.lpush(f"viewer:{viewer_id}:recent_categories", category_id)
        pipe.ltrim(f"viewer:{viewer_id}:recent_categories", 0, RECENT_CATEGORIES_CAP - 1)
        pipe.execute()

        stats_key = f"viewer:{viewer_id}:watch_stats"
        old_ema = self.r.hget(stats_key, "duration_ema")
        new_ema = duration if old_ema is None else (
            EMA_ALPHA * duration + (1 - EMA_ALPHA) * float(old_ema)
        )
        pipe = self.r.pipeline()
        pipe.hset(stats_key, "duration_ema", new_ema)
        pipe.hincrby(stats_key, "session_count", 1)
        pipe.execute()
        self.counts["watch"] += 1

    def handle_chat_event(self, msg):
        if not self._is_new("chat_events", msg["event_id"]):
            return
        stream_id = msg["stream_id"]
        ts = time.time()  # arrival time; good enough for a live velocity signal
        key = f"stream:{stream_id}:chat_ts"
        pipe = self.r.pipeline()
        pipe.zadd(key, {str(msg["event_id"]): ts})
        pipe.zremrangebyscore(key, 0, ts - self.chat_window_sec)
        pipe.expire(key, int(self.chat_window_sec) + 30)
        pipe.execute()
        self.counts["chat"] += 1

    def handle_follow_event(self, msg):
        if not self._is_new("follow_events", msg["event_id"]):
            return
        self.r.sadd(f"viewer:{msg['viewer_id']}:follows", msg["streamer_id"])
        self.counts["follow"] += 1

    def handle_stream_status(self, msg):
        etype = msg["event_type"]
        stream_id = msg["stream_id"]

        if etype == "stream_started":
            if not self._is_new("stream_status", msg["event_id"]):
                return
            self.r.hset(f"stream:{stream_id}:meta", mapping={
                "streamer_id": msg["streamer_id"],
                "category_id": msg["category_id"],
                "language": msg["language"],
                "started_at": msg["ts"],
            })
            self.r.expire(f"stream:{stream_id}:meta", 3600)
            self.counts["stream_started"] += 1

        elif etype == "stream_ended":
            if not self._is_new("stream_status", msg["event_id"]):
                return
            pipe = self.r.pipeline()
            pipe.delete(f"stream:{stream_id}:concurrent_viewers")
            pipe.zrem("live_streams_by_viewers", stream_id)
            pipe.expire(f"stream:{stream_id}:meta", 30)  # brief postmortem window
            pipe.execute()
            self.counts["stream_ended"] += 1

        elif etype == "viewer_count_tick":
            # naturally idempotent overwrite -- no dedup needed
            v = msg["concurrent_viewers"]
            pipe = self.r.pipeline()
            pipe.set(f"stream:{stream_id}:concurrent_viewers", v, ex=CONCURRENT_TTL_SEC)
            pipe.zadd("live_streams_by_viewers", {stream_id: v})
            pipe.execute()
            self.counts["tick"] += 1

    def handle(self, topic, msg):
        if topic == "watch_events":
            self.handle_watch_event(msg)
        elif topic == "chat_events":
            self.handle_chat_event(msg)
        elif topic == "follow_events":
            self.handle_follow_event(msg)
        elif topic == "stream_status":
            self.handle_stream_status(msg)

    def chat_velocity(self, stream_id):
        key = f"stream:{stream_id}:chat_ts"
        now = time.time()
        self.r.zremrangebyscore(key, 0, now - self.chat_window_sec)
        count = self.r.zcard(key)
        return count / (self.chat_window_sec / 60.0)


def compute_lag(consumer):
    assignment = consumer.assignment()
    if not assignment:
        return {}
    end_offsets = consumer.end_offsets(assignment)
    lag_by_topic = {}
    for tp in assignment:
        pos = consumer.position(tp)
        lag_by_topic[tp.topic] = lag_by_topic.get(tp.topic, 0) + max(0, end_offsets[tp] - pos)
    return lag_by_topic


def print_snapshot(fc: FeatureConsumer, r: redis.Redis, consumer, elapsed):
    print(f"\n{'='*78}\n[{elapsed:6.0f}s] SNAPSHOT\n{'='*78}")
    print(f"  processed: watch={fc.counts['watch']} chat={fc.counts['chat']} "
          f"follow={fc.counts['follow']} stream_started={fc.counts['stream_started']} "
          f"stream_ended={fc.counts['stream_ended']} ticks={fc.counts['tick']} "
          f"dupes_skipped={fc.counts['dupes_skipped']}")

    lag = compute_lag(consumer)
    lag_str = ", ".join(f"{t}={l}" for t, l in sorted(lag.items())) if lag else "n/a"
    print(f"  consumer lag: {lag_str}")

    top5 = r.zrevrange("live_streams_by_viewers", 0, 4, withscores=True)
    if not top5:
        print("  no live streams with viewer counts yet")
        return
    print(f"  {'stream_id':>10}  {'streamer':>16}  {'category':>20}  {'concurrent':>10}  {'chat/min':>8}")
    for stream_id_raw, viewers in top5:
        stream_id = int(stream_id_raw)
        meta = r.hgetall(f"stream:{stream_id}:meta")
        streamer_id = meta.get("streamer_id", "?")
        category_id = meta.get("category_id")
        velocity = fc.chat_velocity(stream_id)
        print(f"  {stream_id:>10}  streamer_{str(streamer_id):>6}  category_{str(category_id):>10}  "
              f"{int(viewers):>10}  {velocity:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default="localhost:19092")
    ap.add_argument("--redis-url", default="redis://localhost:6379/0")
    ap.add_argument("--group-id", default="feature-pipeline")
    ap.add_argument("--chat-window-sec", type=float, default=120.0,
                     help="rolling window for chat_velocity (production default would be 300s)")
    ap.add_argument("--snapshot-interval-sec", type=float, default=30.0)
    ap.add_argument("--run-for-sec", type=float, default=None, help="stop after N seconds (default: run forever)")
    args = ap.parse_args()

    r = redis.Redis.from_url(args.redis_url, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {args.redis_url}")

    sc.ensure_topics(args.bootstrap_servers)

    consumer = KafkaConsumer(
        *sc.TOPICS,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    print(f"Subscribed to {sc.TOPICS} as group '{args.group_id}'")
    print(f"chat_velocity window: {args.chat_window_sec}s, snapshot every {args.snapshot_interval_sec}s\n")

    fc = FeatureConsumer(r, args.chat_window_sec)
    start = time.time()
    last_snapshot = start

    try:
        while True:
            records = consumer.poll(timeout_ms=500)
            for tp, batch in records.items():
                for record in batch:
                    fc.handle(tp.topic, record.value)

            now = time.time()
            if now - last_snapshot >= args.snapshot_interval_sec:
                print_snapshot(fc, r, consumer, now - start)
                last_snapshot = now

            if args.run_for_sec and (now - start) >= args.run_for_sec:
                print("\nrun-for-sec reached, stopping.")
                break
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    finally:
        consumer.close()
        print("Consumer closed. Final counts:", fc.counts)


if __name__ == "__main__":
    main()
