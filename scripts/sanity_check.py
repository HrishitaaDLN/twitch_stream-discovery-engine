"""Phase 1 sanity-check report: label distribution, sample rows, summary stats."""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def main():
    watch = load("watch_events")
    streams = load("streams")
    streamers = load("streamers")
    categories = load("categories")
    affinity = load("viewer_category_affinity")
    chat = load("chat_events")
    follows = load("follow_events")
    viewers = load("viewers")

    print("=" * 70)
    print("ROW COUNTS")
    print("=" * 70)
    for name, df in [
        ("categories", categories), ("streamers", streamers), ("viewers", viewers),
        ("viewer_category_affinity", affinity), ("streams", streams),
        ("watch_events", watch), ("chat_events", chat), ("follow_events", follows),
    ]:
        print(f"  {name:28s} {len(df):>10,}")

    print()
    print("=" * 70)
    print("LABEL DISTRIBUTION (watch_events.label_watched_gt_5min)")
    print("=" * 70)
    dist = watch["label_watched_gt_5min"].value_counts(normalize=True).sort_index()
    print(dist.to_string())
    print(f"\n  -> {dist[True]*100:.1f}% positive (watched > 5min)")

    print()
    print("=" * 70)
    print("WATCH_DURATION_SECONDS DISTRIBUTION")
    print("=" * 70)
    print(watch["watch_duration_seconds"].describe().to_string())

    print()
    print("=" * 70)
    print("SAMPLE: 20 random watch_events rows")
    print("=" * 70)
    sample = watch.sample(20, random_state=7).sort_values("event_id")
    print(sample.to_string(index=False))

    print()
    print("=" * 70)
    print("CATEGORY DISTRIBUTION ACROSS STREAMERS (primary_category_id)")
    print("=" * 70)
    cat_names = dict(zip(categories["category_id"], categories["name"]))
    cat_counts = streamers["primary_category_id"].map(cat_names).value_counts()
    cat_pct = (cat_counts / cat_counts.sum() * 100).round(1)
    for name, cnt in cat_counts.items():
        print(f"  {name:24s} {cnt:>4}  ({cat_pct[name]:>5.1f}%)")

    print()
    print("=" * 70)
    print("VIEWER AFFINITY CONCENTRATION")
    print("=" * 70)
    # For each viewer, how many categories hold >=80% of cumulative affinity mass?
    def concentration(group):
        vals = np.sort(group["affinity_score"].values)[::-1]
        cum = np.cumsum(vals)
        total = cum[-1] if len(cum) else 0
        if total == 0:
            return 0
        n80 = int(np.searchsorted(cum, 0.8 * total) + 1)
        return n80

    n80_per_viewer = affinity.groupby("viewer_id").apply(concentration, include_groups=False)
    print("Distribution of 'number of categories needed to cover 80% of a viewer's affinity mass':")
    print(n80_per_viewer.value_counts().sort_index().to_string())
    print(f"\n  Mean categories to reach 80% mass: {n80_per_viewer.mean():.2f}")
    print(f"  Median: {n80_per_viewer.median():.1f}")

    # Also: top-1 category share
    top1_share = affinity.groupby("viewer_id")["affinity_score"].max()
    print(f"\n  Mean top-1 category affinity share: {top1_share.mean():.3f}")
    print(f"  Median top-1 category affinity share: {top1_share.median():.3f}")

    n_cats_per_viewer = affinity.groupby("viewer_id").size()
    print(f"\n  Mean # categories with affinity > 0.005 per viewer: {n_cats_per_viewer.mean():.2f} (of {len(categories)} total)")

    print()
    print("=" * 70)
    print("CHAT / FOLLOW SANITY")
    print("=" * 70)
    print(f"  Chat messages per watch_event (mean): {len(chat) / len(watch):.2f}")
    print(f"  Follow rate per watch_event: {len(follows) / len(watch)*100:.2f}%")
    print(f"  Fraction of streams currently 'live' (ended_at is null): "
          f"{streams['ended_at'].isna().mean()*100:.2f}%")


if __name__ == "__main__":
    main()
