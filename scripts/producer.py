"""
StreamMatch live-activity producer (Phase 2).

Simulates a live platform on top of the EXISTING Phase 1 population
(viewers, streamers, categories, viewer_category_affinity loaded from
Postgres / CSV -- never regenerated here) and publishes events to Kafka
continuously at a configurable rate.

Two event families, published to 4 topics:

1. Stream lifecycle (topic: stream_status)
   - stream_started / stream_ended: a handful of streamers go live and,
     later, end, each run.
   - viewer_count_tick: every --tick-interval-sec per live stream, a
     lifecycle-shaped (ramp-up -> plateau -> decline) concurrent-viewer
     reading, magnitude driven by the streamer's popularity_tier. This is
     the "how hot is this stream right now" signal.

2. Personalization events (topics: watch_events, chat_events, follow_events)
   - Reuses the Phase 1 discovery model: candidate streams are weighted by
     viewer_category_affinity x language match x familiarity (has this
     viewer watched this streamer before) x how hot the stream is right
     now. Watch duration is drawn from the same two-branch
     engaged/bounce sigmoid model as Phase 1, time-compressed so sessions
     resolve in seconds rather than minutes/hours for demo purposes.
   - A watch_event is published once, at session-decision time, carrying
     the full precomputed duration (a simplification vs. modeling
     separate start/heartbeat/end ticks per viewer session).
   - chat_events for that session are published as a burst alongside it,
     with each message's `ts` offset within the session's duration.

Run: python scripts/producer.py --events-per-sec 15 --num-live-streams 15
"""
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone

import numpy as np
from kafka import KafkaProducer

import sim_common as sc

RNG = np.random.default_rng()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def lifecycle_multiplier(frac):
    """0->1 ramp, plateau, 1->0.05 decline, as a function of elapsed/planned."""
    if frac < 0.15:
        return max(0.05, frac / 0.15)
    elif frac < 0.7:
        return 1.0
    else:
        return max(0.05, 1 - (frac - 0.7) / 0.3 * 0.95)


class LiveSim:
    def __init__(self, args, viewers_df, streamers_df, categories_df, affinity_matrix, viewer_id_to_pos):
        self.args = args
        self.viewers_df = viewers_df
        self.streamers_df = streamers_df
        self.categories_df = categories_df
        self.affinity_matrix = affinity_matrix
        self.viewer_id_to_pos = viewer_id_to_pos
        self.n_viewers = len(viewers_df)

        self.cat_names = dict(zip(categories_df["category_id"], categories_df["name"]))
        self.cat_irl = dict(zip(categories_df["category_id"], categories_df["is_irl"]))

        self.viewer_lang = viewers_df["primary_language"].values
        self.viewer_type = viewers_df["viewer_type"].values
        self.viewer_ids = viewers_df["viewer_id"].values

        self.streamer_rows = streamers_df.set_index("streamer_id").to_dict("index")
        self.streamer_ids = list(self.streamer_rows.keys())

        self.live_streams = {}       # stream_id -> state dict
        self._next_stream_id = 10_000_000
        self._next_event_id = 1

        self.familiarity = {}        # (viewer_pos, streamer_id) -> count
        self.follows = set()         # (viewer_pos, streamer_id)

        self._session_credit = 0.0
        self._last_loop_time = time.time()

        self.stats = {"watch": 0, "chat": 0, "follow": 0, "stream_started": 0, "stream_ended": 0, "ticks": 0}

    def next_id(self):
        self._next_event_id += 1
        return self._next_event_id

    def next_stream_id(self):
        self._next_stream_id += 1
        return self._next_stream_id

    # -- stream lifecycle -------------------------------------------------

    def spawn_stream(self, seed_elapsed_frac=None):
        streamer_id = random.choice(self.streamer_ids)
        s = self.streamer_rows[streamer_id]
        tier = s["popularity_tier"]
        lo, hi = sc.TIER_PEAK_RANGE[tier]
        peak = float(RNG.uniform(lo, hi))

        use_secondary = s.get("secondary_category_id") not in (None,) and not (
            isinstance(s.get("secondary_category_id"), float) and math.isnan(s["secondary_category_id"])
        ) and RNG.random() < 0.15
        category_id = int(s["secondary_category_id"]) if use_secondary else int(s["primary_category_id"])

        duration = RNG.uniform(self.args.stream_duration_min, self.args.stream_duration_max)
        start_time = time.time()
        if seed_elapsed_frac is not None:
            start_time -= seed_elapsed_frac * duration

        stream_id = self.next_stream_id()
        self.live_streams[stream_id] = {
            "streamer_id": streamer_id,
            "username": s["username"],
            "category_id": category_id,
            "language": s["language"],
            "tier": tier,
            "peak": peak,
            "start_time": start_time,
            "planned_duration": duration,
            "current_viewers": max(1, peak * 0.1),
            "last_tick": 0.0,
        }
        return stream_id

    def publish_stream_started(self, producer, stream_id):
        st = self.live_streams[stream_id]
        producer.send("stream_status", key=str(stream_id), value={
            "event_type": "stream_started",
            "event_id": self.next_id(),
            "stream_id": stream_id,
            "streamer_id": st["streamer_id"],
            "category_id": st["category_id"],
            "language": st["language"],
            "ts": now_iso(),
        })
        self.stats["stream_started"] += 1

    def publish_stream_ended(self, producer, stream_id):
        producer.send("stream_status", key=str(stream_id), value={
            "event_type": "stream_ended",
            "event_id": self.next_id(),
            "stream_id": stream_id,
            "ts": now_iso(),
        })
        self.stats["stream_ended"] += 1

    def publish_tick(self, producer, stream_id):
        st = self.live_streams[stream_id]
        elapsed = time.time() - st["start_time"]
        frac = min(1.0, elapsed / st["planned_duration"])
        noise = float(RNG.lognormal(0, 0.15))
        current = max(1, round(st["peak"] * lifecycle_multiplier(frac) * noise))
        st["current_viewers"] = current
        producer.send("stream_status", key=str(stream_id), value={
            "event_type": "viewer_count_tick",
            "event_id": self.next_id(),
            "stream_id": stream_id,
            "concurrent_viewers": current,
            "ts": now_iso(),
        })
        self.stats["ticks"] += 1

    def tick_streams(self, producer):
        now = time.time()
        ended = []
        for stream_id, st in self.live_streams.items():
            elapsed = now - st["start_time"]
            if elapsed >= st["planned_duration"]:
                ended.append(stream_id)
                continue
            if now - st["last_tick"] >= self.args.tick_interval_sec:
                self.publish_tick(producer, stream_id)
                st["last_tick"] = now

        for stream_id in ended:
            self.publish_stream_ended(producer, stream_id)
            del self.live_streams[stream_id]

        while len(self.live_streams) < self.args.num_live_streams:
            stream_id = self.spawn_stream()
            self.publish_stream_started(producer, stream_id)
            self.publish_tick(producer, stream_id)  # immediate first reading

    # -- personalization sessions ------------------------------------------

    def pick_stream_for_viewer(self, vpos):
        if not self.live_streams:
            return None
        stream_ids = list(self.live_streams.keys())
        cats = np.array([self.live_streams[sid]["category_id"] for sid in stream_ids])
        langs = np.array([self.live_streams[sid]["language"] for sid in stream_ids])
        peaks = np.array([self.live_streams[sid]["current_viewers"] for sid in stream_ids], dtype=float)
        streamer_ids = np.array([self.live_streams[sid]["streamer_id"] for sid in stream_ids])

        cat_pos = np.array([sc_pos for sc_pos in cats]) - 1  # category_id is 1-indexed and dense
        aff = self.affinity_matrix[vpos, cat_pos]
        lang_match = (langs == self.viewer_lang[vpos]).astype(float)
        fam_counts = np.array([self.familiarity.get((vpos, sid), 0) for sid in streamer_ids], dtype=float)
        followed = np.array([(vpos, sid) in self.follows for sid in streamer_ids], dtype=float)

        fam_mult = 1.0 + np.minimum(fam_counts, 6) * 0.9 + followed * 1.5
        lang_mult = np.where(lang_match > 0, 3.0, 1.0)
        pop_mult = 1.0 + np.log1p(peaks) * 0.4

        weights = (aff + 0.03) * lang_mult * fam_mult * pop_mult
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()
        idx = RNG.choice(len(stream_ids), p=probs)
        return stream_ids[idx], float(aff[idx]), float(lang_match[idx])

    def run_session(self, producer, vpos):
        picked = self.pick_stream_for_viewer(vpos)
        if picked is None:
            return
        stream_id, aff, lang_match = picked
        st = self.live_streams[stream_id]
        streamer_id = st["streamer_id"]
        category_id = st["category_id"]
        is_irl = bool(self.cat_irl.get(category_id, False))

        fam = self.familiarity.get((vpos, streamer_id), 0)
        is_followed = (vpos, streamer_id) in self.follows
        quality_z = float(RNG.normal(0, 1))

        logit = (
            -1.55
            + 2.6 * aff
            + 0.55 * lang_match
            + 0.30 * min(fam, 5)
            + 0.55 * float(is_followed)
            + 0.45 * quality_z
        )
        engaged = RNG.random() < sc.sigmoid(logit)

        # time-compressed durations so sessions resolve in seconds, not minutes/hours
        if engaged:
            duration = float(np.clip(RNG.lognormal(math.log(22), 0.8), 2, 240))
        else:
            duration = float(np.clip(RNG.lognormal(math.log(3), 0.9), 1, 60))
        duration_s = int(round(duration))

        viewer_id = int(self.viewer_ids[vpos])
        session_start = time.time()
        event_id = self.next_id()
        producer.send("watch_events", key=str(viewer_id), value={
            "event_id": event_id,
            "viewer_id": viewer_id,
            "stream_id": stream_id,
            "streamer_id": streamer_id,
            "category_id": category_id,
            "session_start": now_iso(),
            "watch_duration_seconds": duration_s,
            "label_watched_gt_5min": duration_s > 300,
        })
        self.stats["watch"] += 1

        # chat burst for this session
        chat_lambda_base = 1.0 if not is_irl else 2.0
        type_mult = {"casual": 0.6, "regular": 1.0, "power": 2.0}[self.viewer_type[vpos]]
        duration_mult = min(duration_s / 20.0, 6.0)
        chat_lambda = chat_lambda_base * type_mult * duration_mult
        n_chats = RNG.poisson(chat_lambda)
        if n_chats > 0:
            phrase_pool = sc.GENERAL_PHRASES + (sc.IRL_PHRASES if is_irl else sc.GAME_PHRASES)
            offsets = np.sort(RNG.uniform(0, max(duration_s, 1), size=n_chats))
            for off in offsets:
                producer.send("chat_events", key=str(stream_id), value={
                    "event_id": self.next_id(),
                    "viewer_id": viewer_id,
                    "stream_id": stream_id,
                    "message_text": random.choice(phrase_pool),
                    "ts": datetime.fromtimestamp(session_start + off, tz=timezone.utc).isoformat(),
                })
                self.stats["chat"] += 1

        new_fam = fam + 1
        if not is_followed and new_fam >= 2 and engaged:
            follow_p = min(0.50 + 0.28 * new_fam, 0.9)
            if RNG.random() < follow_p:
                self.follows.add((vpos, streamer_id))
                producer.send("follow_events", key=str(viewer_id), value={
                    "event_id": self.next_id(),
                    "viewer_id": viewer_id,
                    "streamer_id": streamer_id,
                    "followed_at": now_iso(),
                })
                self.stats["follow"] += 1
        self.familiarity[(vpos, streamer_id)] = new_fam

    def run_session_batch(self, producer, dt):
        self._session_credit += self.args.events_per_sec * dt
        n = int(self._session_credit)
        self._session_credit -= n
        for _ in range(n):
            vpos = RNG.integers(0, self.n_viewers)
            self.run_session(producer, vpos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default="localhost:19092")
    ap.add_argument("--events-per-sec", type=float, default=15.0, help="target watch_events/sec")
    ap.add_argument("--num-live-streams", type=int, default=15)
    ap.add_argument("--tick-interval-sec", type=float, default=2.0)
    ap.add_argument("--stream-duration-min", type=float, default=60.0, help="seconds (compressed timeline)")
    ap.add_argument("--stream-duration-max", type=float, default=180.0, help="seconds (compressed timeline)")
    ap.add_argument("--seed-live-streams", type=int, default=8, help="streams already mid-lifecycle at t=0")
    ap.add_argument("--run-for-sec", type=float, default=None, help="stop after N seconds (default: run forever)")
    args = ap.parse_args()

    print("Loading population...")
    viewers_df, streamers_df, categories_df, affinity_matrix, viewer_id_to_pos, cat_id_to_pos = sc.load_population()

    print(f"Ensuring Kafka topics exist on {args.bootstrap_servers}...")
    sc.ensure_topics(args.bootstrap_servers)

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        linger_ms=20,
        acks=1,
    )

    sim = LiveSim(args, viewers_df, streamers_df, categories_df, affinity_matrix, viewer_id_to_pos)

    print(f"Seeding {args.seed_live_streams} streams already mid-lifecycle...")
    for _ in range(args.seed_live_streams):
        stream_id = sim.spawn_stream(seed_elapsed_frac=float(RNG.uniform(0.1, 0.6)))
        sim.publish_stream_started(producer, stream_id)
        sim.publish_tick(producer, stream_id)

    print(f"Running live simulation: {args.events_per_sec} watch_events/sec target, "
          f"{args.num_live_streams} steady-state live streams, tick every {args.tick_interval_sec}s")
    print("Ctrl+C to stop.\n")

    start = time.time()
    last_report = start
    try:
        while True:
            loop_start = time.time()
            dt = loop_start - sim._last_loop_time
            sim._last_loop_time = loop_start

            sim.tick_streams(producer)
            sim.run_session_batch(producer, dt)

            if loop_start - last_report >= 10:
                elapsed = loop_start - start
                print(f"[{elapsed:6.0f}s] live_streams={len(sim.live_streams):3d} "
                      f"watch={sim.stats['watch']:6d} chat={sim.stats['chat']:6d} "
                      f"follow={sim.stats['follow']:5d} started={sim.stats['stream_started']:4d} "
                      f"ended={sim.stats['stream_ended']:4d} ticks={sim.stats['ticks']:5d}")
                last_report = loop_start

            if args.run_for_sec and (loop_start - start) >= args.run_for_sec:
                print("run-for-sec reached, stopping.")
                break

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed. Final stats:", sim.stats)


if __name__ == "__main__":
    main()
