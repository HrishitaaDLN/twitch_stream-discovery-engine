"""
StreamMatch synthetic data generator (Phase 1).

Generates correlated, realistic Twitch-like data:
  categories, streamers, viewers, viewer_category_affinity, streams,
  watch_events, chat_events, follow_events

Design notes:
- viewer_category_affinity is Dirichlet-sampled per viewer so preference mass
  concentrates on a few categories, not spread uniformly.
- Stream discovery (which stream a viewer's session lands on) is weighted by
  affinity x language match x familiarity (repeat viewership) x how "hot"
  the stream is right now -- a simplified stand-in for the real recsys
  problem, baked into the generator so downstream features have real signal.
- watch_duration_seconds is generated from a two-branch (engaged / bounce)
  model driven by the same signals, so the 5-minute label is learnable but
  not deterministic from any single feature.
- Everything is processed in a single global time-ordered pass so
  "has this viewer watched this streamer before" / "does this viewer follow
  this streamer" are always causally correct (no lookahead).

Output: CSVs in ../data, one per table, matching sql/schema.sql exactly.
"""

import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

N_STREAMERS = 400
N_VIEWERS = 5000
SIM_DAYS = 30
SIM_START = datetime(2026, 7, 28, 0, 0, 0)
SIM_END = SIM_START + timedelta(days=SIM_DAYS)
SIM_START_EPOCH = SIM_START.timestamp()
SIM_END_EPOCH = SIM_END.timestamp()

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

CATEGORIES = [
    ("Just Chatting", True),
    ("League of Legends", False),
    ("VALORANT", False),
    ("Grand Theft Auto V", False),
    ("Fortnite", False),
    ("Counter-Strike 2", False),
    ("Minecraft", False),
    ("Apex Legends", False),
    ("Call of Duty: Warzone", False),
    ("Dota 2", False),
    ("World of Warcraft", False),
    ("Overwatch 2", False),
    ("IRL", True),
    ("Sports", True),
    ("Music", True),
    ("Poker", False),
    ("Chess", False),
    ("Art", True),
]
N_CATEGORIES = len(CATEGORIES)

# Zipf-ish popularity weight by rank (index 0 = most popular)
CATEGORY_POPULARITY = np.array([1.0 / ((i + 1) ** 0.8) for i in range(N_CATEGORIES)])
CATEGORY_POPULARITY /= CATEGORY_POPULARITY.sum()

LANGUAGES = ["en", "es", "pt", "ko", "fr", "de", "ja"]
LANGUAGE_WEIGHTS = [0.55, 0.12, 0.08, 0.07, 0.06, 0.06, 0.06]

POPULARITY_TIERS = ["micro", "small", "mid", "large", "mega"]
POPULARITY_TIER_WEIGHTS = [0.70, 0.20, 0.07, 0.025, 0.005]
TIER_PEAK_RANGE = {
    "micro": (1, 15),
    "small": (15, 60),
    "mid": (60, 300),
    "large": (300, 1500),
    "mega": (1500, 8000),
}
TIER_STREAMS_LAMBDA = {  # expected number of streams over SIM_DAYS
    "micro": 6,
    "small": 9,
    "mid": 14,
    "large": 20,
    "mega": 26,
}

VIEWER_TYPES = ["casual", "regular", "power"]
VIEWER_TYPE_WEIGHTS = [0.60, 0.30, 0.10]
VIEWER_TYPE_DAILY_LAMBDA = {"casual": 0.5, "regular": 1.5, "power": 3.0}

GENERAL_PHRASES = [
    "LOL", "PogChamp", "let's go!!", "no way", "clip that", "W stream",
    "hi chat", "insane", "hahaha", "GG", "first time here, this is great",
    "moderators pls", "monkaS", "based", "real", "sheesh",
]
GAME_PHRASES = [
    "nice play", "how'd you do that", "clutch!!", "ez", "rigged lol",
    "carry me", "what rank are you", "insane aim", "throw", "1v1 me",
]
IRL_PHRASES = [
    "hi from Germany", "what music is this", "camera guy is doing great",
    "love this", "first stream I've caught live", "hydrate!", "so real",
]


def out_path(name):
    return os.path.join(OUT_DIR, f"{name}.csv")


# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------

def gen_categories():
    rows = []
    for i, (name, is_irl) in enumerate(CATEGORIES, start=1):
        rows.append({"category_id": i, "name": name, "is_irl": is_irl})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Streamers
# --------------------------------------------------------------------------

def gen_streamers():
    rows = []
    tiers = RNG.choice(POPULARITY_TIERS, size=N_STREAMERS, p=POPULARITY_TIER_WEIGHTS)
    primary_cats = RNG.choice(np.arange(1, N_CATEGORIES + 1), size=N_STREAMERS, p=CATEGORY_POPULARITY)
    langs = RNG.choice(LANGUAGES, size=N_STREAMERS, p=LANGUAGE_WEIGHTS)
    has_secondary = RNG.random(N_STREAMERS) < 0.20

    peak_baseline = np.zeros(N_STREAMERS)
    for i in range(N_STREAMERS):
        lo, hi = TIER_PEAK_RANGE[tiers[i]]
        peak_baseline[i] = RNG.uniform(lo, hi)

    created = [
        SIM_START - timedelta(days=int(RNG.uniform(30, 730)))
        for _ in range(N_STREAMERS)
    ]

    for i in range(N_STREAMERS):
        secondary = None
        if has_secondary[i]:
            cand = int(RNG.choice(np.arange(1, N_CATEGORIES + 1), p=CATEGORY_POPULARITY))
            tries = 0
            while cand == primary_cats[i] and tries < 5:
                cand = int(RNG.choice(np.arange(1, N_CATEGORIES + 1), p=CATEGORY_POPULARITY))
                tries += 1
            secondary = cand if cand != primary_cats[i] else None
        rows.append({
            "streamer_id": i + 1,
            "username": f"streamer_{i+1:04d}",
            "primary_category_id": int(primary_cats[i]),
            "secondary_category_id": secondary,
            "language": langs[i],
            "popularity_tier": tiers[i],
            "account_created_at": created[i],
            "_peak_baseline": peak_baseline[i],  # internal only, not exported
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Viewers + affinity
# --------------------------------------------------------------------------

def gen_viewers():
    langs = RNG.choice(LANGUAGES, size=N_VIEWERS, p=LANGUAGE_WEIGHTS)
    types = RNG.choice(VIEWER_TYPES, size=N_VIEWERS, p=VIEWER_TYPE_WEIGHTS)
    signup = [SIM_START - timedelta(days=int(RNG.uniform(1, 730))) for _ in range(N_VIEWERS)]
    pref_hour = RNG.normal(20, 4, size=N_VIEWERS) % 24

    df = pd.DataFrame({
        "viewer_id": np.arange(1, N_VIEWERS + 1),
        "username": [f"viewer_{i+1:05d}" for i in range(N_VIEWERS)],
        "primary_language": langs,
        "viewer_type": types,
        "signup_at": signup,
    })
    return df, pref_hour


def gen_affinity(viewers_df):
    # Concentrated Dirichlet: alpha scaled down so most mass lands on a few
    # categories, weighted toward globally popular categories.
    alpha = CATEGORY_POPULARITY * 3.0  # small alphas -> concentrated draws
    affinity = RNG.dirichlet(alpha, size=N_VIEWERS)  # (N_VIEWERS, N_CATEGORIES)

    rows = []
    for v in range(N_VIEWERS):
        for c in range(N_CATEGORIES):
            score = affinity[v, c]
            if score > 0.005:  # skip near-zero noise rows
                rows.append({
                    "viewer_id": v + 1,
                    "category_id": c + 1,
                    "affinity_score": float(score),
                })
    return pd.DataFrame(rows), affinity  # also return dense matrix for gen speed


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------

def gen_streams(streamers_df):
    rows = []
    stream_id = 1
    hour_weights = np.exp(-0.5 * ((np.arange(24) - 20) / 4.0) ** 2) + 0.15
    hour_weights /= hour_weights.sum()

    for _, s in streamers_df.iterrows():
        lam = TIER_STREAMS_LAMBDA[s["popularity_tier"]]
        n_streams = max(1, RNG.poisson(lam))
        for _ in range(n_streams):
            day_offset = RNG.integers(0, SIM_DAYS)
            hour = RNG.choice(24, p=hour_weights)
            minute = RNG.integers(0, 60)
            started_at = SIM_START + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute))

            dur_hours = float(np.clip(RNG.lognormal(math.log(3.0), 0.4), 1.0, 8.0))
            computed_end = started_at + timedelta(hours=dur_hours)

            if computed_end <= SIM_END:
                ended_at = computed_end
                effective_end_epoch = computed_end.timestamp()
            else:
                ended_at = None  # still live at snapshot time
                effective_end_epoch = SIM_END_EPOCH

            use_secondary = (
                pd.notna(s["secondary_category_id"]) and RNG.random() < 0.15
            )
            category_id = int(s["secondary_category_id"]) if use_secondary else int(s["primary_category_id"])

            peak = float(s["_peak_baseline"] * RNG.lognormal(0, 0.35))
            peak = max(peak, 1.0)

            rows.append({
                "stream_id": stream_id,
                "streamer_id": int(s["streamer_id"]),
                "category_id": category_id,
                "title": f"{s['username']} - {CATEGORIES[category_id-1][0]}",
                "language": s["language"],
                "started_at": started_at,
                "ended_at": ended_at,
                "_start_epoch": started_at.timestamp(),
                "_end_epoch": effective_end_epoch,
                "_peak": peak,
            })
            stream_id += 1

    return pd.DataFrame(rows)


def build_hourly_index(streams_df):
    """hour_idx -> np.array of row-positions in streams_df live during that hour."""
    buckets = {}
    starts = streams_df["_start_epoch"].values
    ends = streams_df["_end_epoch"].values
    for pos in range(len(streams_df)):
        h0 = int((starts[pos] - SIM_START_EPOCH) // 3600)
        h1 = int((ends[pos] - SIM_START_EPOCH) // 3600)
        for h in range(max(h0, 0), h1 + 1):
            buckets.setdefault(h, []).append(pos)
    return {h: np.array(v, dtype=np.int64) for h, v in buckets.items()}


# --------------------------------------------------------------------------
# Sessions (watch/chat/follow) -- single time-ordered pass
# --------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gen_sessions(viewers_df, pref_hour, affinity_matrix, streamers_df, streams_df, hourly_index):
    # 1. Sample session timestamps per viewer (independent of stream choice)
    session_list = []  # (epoch, viewer_pos)
    for v in range(N_VIEWERS):
        vtype = viewers_df.at[v, "viewer_type"]
        lam = VIEWER_TYPE_DAILY_LAMBDA[vtype] * SIM_DAYS
        n_sessions = RNG.poisson(lam)
        if n_sessions == 0:
            continue
        days = RNG.integers(0, SIM_DAYS, size=n_sessions)
        hours = np.clip(RNG.normal(pref_hour[v], 2.5, size=n_sessions), 0, 23.99)
        for d, h in zip(days, hours):
            ts = SIM_START_EPOCH + d * 86400 + h * 3600 + RNG.integers(0, 3600)
            session_list.append((ts, v))

    session_list.sort(key=lambda x: x[0])

    # Fast lookup arrays for streams
    s_streamer = streams_df["streamer_id"].values
    s_category = streams_df["category_id"].values
    s_language = streams_df["language"].values
    s_start = streams_df["_start_epoch"].values
    s_end = streams_df["_end_epoch"].values
    s_peak = streams_df["_peak"].values
    s_stream_id = streams_df["stream_id"].values
    streamer_lang = dict(zip(streamers_df["streamer_id"], streamers_df["language"]))

    viewer_lang = viewers_df["primary_language"].values
    viewer_type_arr = viewers_df["viewer_type"].values

    familiarity = {}   # (viewer_pos, streamer_id) -> count
    follows = set()    # (viewer_pos, streamer_id)
    last_engaged_quality = {}

    watch_rows = []
    chat_rows = []
    follow_rows = []
    skipped = 0

    watch_id = 1
    chat_id = 1
    follow_id = 1

    for ts, v in session_list:
        hour_idx = int((ts - SIM_START_EPOCH) // 3600)
        cand_pos = hourly_index.get(hour_idx)
        if cand_pos is None or len(cand_pos) == 0:
            skipped += 1
            continue

        starts = s_start[cand_pos]
        ends = s_end[cand_pos]
        live_mask = (starts <= ts) & (ts < ends)
        cand_pos = cand_pos[live_mask]
        if len(cand_pos) == 0:
            skipped += 1
            continue

        cats = s_category[cand_pos]
        langs = s_language[cand_pos]
        peaks = s_peak[cand_pos]
        streamer_ids = s_streamer[cand_pos]

        aff = affinity_matrix[v, cats - 1]
        lang_match = (langs == viewer_lang[v]).astype(float)
        fam_counts = np.array([familiarity.get((v, sid), 0) for sid in streamer_ids], dtype=float)
        followed_flags = np.array([(v, sid) in follows for sid in streamer_ids], dtype=float)

        fam_mult = 1.0 + np.minimum(fam_counts, 6) * 0.9 + followed_flags * 1.5
        lang_mult = np.where(lang_match > 0, 3.0, 1.0)
        pop_mult = 1.0 + np.log1p(peaks) * 0.4

        weights = (aff + 0.03) * lang_mult * fam_mult * pop_mult
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()

        choice_idx = RNG.choice(len(cand_pos), p=probs)
        pos = cand_pos[choice_idx]
        streamer_id = int(s_streamer[pos])
        stream_id = int(s_stream_id[pos])
        category_id = int(s_category[pos])
        is_irl = CATEGORIES[category_id - 1][1]

        fam = familiarity.get((v, streamer_id), 0)
        is_followed = (v, streamer_id) in follows
        stream_quality_z = float(RNG.normal(0, 1))

        logit = (
            -1.55
            + 2.6 * float(aff[choice_idx])
            + 0.55 * float(lang_match[choice_idx])
            + 0.30 * min(fam, 5)
            + 0.55 * float(is_followed)
            + 0.45 * stream_quality_z
        )
        p_engaged = sigmoid(logit)
        engaged = RNG.random() < p_engaged

        remaining = max(5.0, s_end[pos] - ts)
        if engaged:
            duration = RNG.lognormal(math.log(1400), 0.8)
        else:
            duration = RNG.lognormal(math.log(70), 0.9)
        duration = float(np.clip(duration, 10, remaining))
        duration_s = int(duration)

        watch_rows.append({
            "event_id": watch_id,
            "viewer_id": v + 1,
            "stream_id": stream_id,
            "session_start": datetime.fromtimestamp(ts),
            "watch_duration_seconds": duration_s,
            "label_watched_gt_5min": duration_s > 300,
        })

        # chat generation
        chat_lambda_base = 1.0 if not is_irl else 2.0
        type_mult = {"casual": 0.6, "regular": 1.0, "power": 2.0}[viewer_type_arr[v]]
        duration_mult = min(duration_s / 300.0, 6.0)
        chat_lambda = chat_lambda_base * type_mult * duration_mult
        n_chats = RNG.poisson(chat_lambda)
        if n_chats > 0:
            phrase_pool = GENERAL_PHRASES + (IRL_PHRASES if is_irl else GAME_PHRASES)
            offsets = RNG.uniform(0, duration_s, size=n_chats)
            offsets.sort()
            for off in offsets:
                chat_rows.append({
                    "event_id": chat_id,
                    "viewer_id": v + 1,
                    "stream_id": stream_id,
                    "message_text": random.choice(phrase_pool),
                    "ts": datetime.fromtimestamp(ts + off),
                })
                chat_id += 1

        # follow decision (causal: only affects *future* sessions)
        new_fam = fam + 1
        if not is_followed and new_fam >= 2 and engaged:
            follow_p = min(0.50 + 0.28 * new_fam, 0.9)
            if RNG.random() < follow_p:
                follows.add((v, streamer_id))
                follow_rows.append({
                    "event_id": follow_id,
                    "viewer_id": v + 1,
                    "streamer_id": streamer_id,
                    "followed_at": datetime.fromtimestamp(ts + duration),
                })
                follow_id += 1

        familiarity[(v, streamer_id)] = new_fam
        watch_id += 1

    print(f"  sessions requested: {len(session_list)}, skipped (no live candidates): {skipped}")
    return (
        pd.DataFrame(watch_rows),
        pd.DataFrame(chat_rows),
        pd.DataFrame(follow_rows),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating categories...")
    categories_df = gen_categories()

    print("Generating streamers...")
    streamers_df = gen_streamers()

    print("Generating viewers...")
    viewers_df, pref_hour = gen_viewers()

    print("Generating viewer_category_affinity...")
    affinity_df, affinity_matrix = gen_affinity(viewers_df)

    print("Generating streams...")
    streams_df = gen_streams(streamers_df)

    print("Building hourly live-stream index...")
    hourly_index = build_hourly_index(streams_df)

    print("Generating watch/chat/follow events (single causal pass)...")
    watch_df, chat_df, follow_df = gen_sessions(
        viewers_df, pref_hour, affinity_matrix, streamers_df, streams_df, hourly_index
    )

    # Strip internal-only columns before export
    streamers_export = streamers_df.drop(columns=["_peak_baseline"])
    streams_export = streams_df.drop(columns=["_start_epoch", "_end_epoch", "_peak"])

    categories_df.to_csv(out_path("categories"), index=False)
    streamers_export.to_csv(out_path("streamers"), index=False)
    viewers_df.to_csv(out_path("viewers"), index=False)
    affinity_df.to_csv(out_path("viewer_category_affinity"), index=False)
    streams_export.to_csv(out_path("streams"), index=False)
    watch_df.to_csv(out_path("watch_events"), index=False)
    chat_df.to_csv(out_path("chat_events"), index=False)
    follow_df.to_csv(out_path("follow_events"), index=False)

    print("\nRow counts:")
    for name, df in [
        ("categories", categories_df), ("streamers", streamers_export),
        ("viewers", viewers_df), ("viewer_category_affinity", affinity_df),
        ("streams", streams_export), ("watch_events", watch_df),
        ("chat_events", chat_df), ("follow_events", follow_df),
    ]:
        print(f"  {name}: {len(df):,}")


if __name__ == "__main__":
    main()
