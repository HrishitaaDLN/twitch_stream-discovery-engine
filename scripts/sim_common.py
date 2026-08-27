"""Shared constants and helpers for the Phase 2 live producer.

Reuses the same category/tier tables and correlation logic as Phase 1's
generate_data.py, but does NOT regenerate a population -- it loads the
existing viewers/streamers/categories/affinity from Postgres (falling back
to the Phase 1 CSVs if Postgres isn't reachable) and simulates live
behavior on top of it.
"""

import os
import math
import numpy as np
import pandas as pd

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://streammatch:streammatch@localhost:5433/streammatch"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TIER_PEAK_RANGE = {
    "micro": (1, 15),
    "small": (15, 60),
    "mid": (60, 300),
    "large": (300, 1500),
    "mega": (1500, 8000),
}

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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_population():
    """Returns (viewers_df, streamers_df, categories_df, affinity_matrix, cat_id_to_pos).

    affinity_matrix is a dense (n_viewers, n_categories) numpy array indexed
    by position (0-based), not by viewer_id/category_id directly -- use the
    returned id->pos maps.
    """
    source = None
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
        viewers_df = pd.read_sql("SELECT * FROM viewers ORDER BY viewer_id", conn)
        streamers_df = pd.read_sql("SELECT * FROM streamers ORDER BY streamer_id", conn)
        categories_df = pd.read_sql("SELECT * FROM categories ORDER BY category_id", conn)
        affinity_df = pd.read_sql("SELECT * FROM viewer_category_affinity", conn)
        conn.close()
        source = "postgres"
    except Exception as e:
        print(f"[sim_common] Postgres unavailable ({e}); falling back to CSVs in {DATA_DIR}")
        viewers_df = pd.read_csv(os.path.join(DATA_DIR, "viewers.csv"))
        streamers_df = pd.read_csv(os.path.join(DATA_DIR, "streamers.csv"))
        categories_df = pd.read_csv(os.path.join(DATA_DIR, "categories.csv"))
        affinity_df = pd.read_csv(os.path.join(DATA_DIR, "viewer_category_affinity.csv"))
        source = "csv"

    print(f"[sim_common] loaded population from {source}: "
          f"{len(viewers_df)} viewers, {len(streamers_df)} streamers, {len(categories_df)} categories")

    viewer_id_to_pos = {vid: i for i, vid in enumerate(viewers_df["viewer_id"])}
    cat_id_to_pos = {cid: i for i, cid in enumerate(categories_df["category_id"])}
    n_viewers = len(viewers_df)
    n_categories = len(categories_df)

    affinity_matrix = np.full((n_viewers, n_categories), 0.03 / n_categories, dtype=float)
    for row in affinity_df.itertuples(index=False):
        vpos = viewer_id_to_pos.get(row.viewer_id)
        cpos = cat_id_to_pos.get(row.category_id)
        if vpos is not None and cpos is not None:
            affinity_matrix[vpos, cpos] = row.affinity_score

    return viewers_df, streamers_df, categories_df, affinity_matrix, viewer_id_to_pos, cat_id_to_pos


TOPICS = ["watch_events", "chat_events", "follow_events", "stream_status"]


def ensure_topics(bootstrap_servers, num_partitions=3):
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id="streammatch-admin")
    try:
        topics = [NewTopic(name=t, num_partitions=num_partitions, replication_factor=1) for t in TOPICS]
        admin.create_topics(topics)
        print(f"[sim_common] created topics: {TOPICS}")
    except TopicAlreadyExistsError:
        print(f"[sim_common] topics already exist: {TOPICS}")
    finally:
        admin.close()
