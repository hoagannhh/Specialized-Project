"""Triplets from an already partitioned source pool; CLI uses the shared builder."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sample_negatives(df, strategy="mixed", seed=42, chunk_size=512,
                     top_k_pool=5, vectorizer=None):
    """Mine lexical candidates in blocks, always excluding the whole source group.

    Mixed requests hard negatives for half the rows. Zero-overlap/empty blocks
    fall back to valid random negatives from this split. Hard candidates are
    heuristic non-matches, not manually verified semantic negatives.
    """
    if strategy not in {"random", "hard", "mixed"}:
        raise ValueError("strategy must be random, hard or mixed")
    if chunk_size < 2 or top_k_pool < 1:
        raise ValueError("chunk_size >= 2 and top_k_pool >= 1 required")
    groups = df["group_id"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Need at least two source groups in each split")
    rng = np.random.default_rng(seed)
    n = len(df)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranges = {sorted_groups[s]: (s, e) for s, e in zip(starts, ends)}
    selected = np.empty(n, dtype=np.int64)
    kinds = np.full(n, "random", dtype=object)
    for i, group in enumerate(groups):
        start, end = ranges[group]
        j = rng.integers(n - (end - start))
        selected[i] = order[j if j < start else j + end - start]
    if strategy != "random":
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=100000, dtype=np.float32)
            vectorizer.fit(pd.concat([df.original_text, df.paraphrase_text]))
        indices = rng.permutation(n)
        for start in range(0, n, chunk_size):
            block = indices[start:start + chunk_size]
            scores = cosine_similarity(
                vectorizer.transform(df.original_text.iloc[block]),
                vectorizer.transform(df.paraphrase_text.iloc[block]),
            )
            for local, i in enumerate(block):
                if strategy == "mixed" and rng.random() >= 0.5:
                    continue
                eligible = np.flatnonzero((groups[block] != groups[i]) & (scores[local] > 0))
                if not len(eligible):
                    continue
                best = eligible[np.argsort(scores[local, eligible])[-top_k_pool:]]
                selected[i] = block[rng.choice(best)]
                kinds[i] = "lexical_hard"
    assert np.all(groups[selected] != groups)
    return selected, kinds


def make_triplets(df, selected, kinds):
    df = df.reset_index(drop=True)
    neg = df.iloc[selected].reset_index(drop=True)
    result = pd.DataFrame({
        "anchor": df.original_text, "positive": df.paraphrase_text,
        "negative": neg.paraphrase_text, "group_id": df.group_id,
        "negative_group_id": neg.group_id, "negative_type": kinds,
        "source": "humarin",
    })
    if (result.group_id == result.negative_group_id).any():
        raise ValueError("Negative from the anchor's source component")
    if ((result.negative == result.positive) | (result.negative == result.anchor)).any():
        raise ValueError("Negative equals anchor/positive")
    return result


if __name__ == "__main__":
    try:
        from .prepare_datasets import main
    except ImportError:
        from prepare_datasets import main
    main()
