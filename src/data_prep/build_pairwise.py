"""Pair labels describe paraphrase relationships, not authorship/AI labels."""
import pandas as pd


def make_pairwise(triplets, seed=42):
    positive = pd.DataFrame({
        "sentence_1": triplets.anchor, "sentence_2": triplets.positive,
        "label": 1, "group_id": triplets.group_id,
        "candidate_group_id": triplets.group_id, "pair_type": "positive",
        "source": triplets.source,
    })
    negative = pd.DataFrame({
        "sentence_1": triplets.anchor, "sentence_2": triplets.negative,
        "label": 0, "group_id": triplets.group_id,
        "candidate_group_id": triplets.negative_group_id,
        "pair_type": triplets.negative_type, "source": triplets.source,
    })
    result = pd.concat([positive, negative], ignore_index=True)
    result = result.drop_duplicates(["sentence_1", "sentence_2", "label"])
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    try:
        from .prepare_datasets import main
    except ImportError:
        from prepare_datasets import main
    main()
