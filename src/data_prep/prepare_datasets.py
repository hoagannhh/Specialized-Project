"""Build three training views from one source-component split, without downloading.

Existing splits are archived only after a successful build. PADBen per-sentence
labels are remapped to AI-touched; they are never used as pair-relation labels.
"""
import argparse
import hashlib
import json
import sys
import uuid
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .audit_datasets import audit
    from .common import load_humarin_pairs, RAW_DIR, PROJECT_ROOT
    from .build_triplet import sample_negatives, make_triplets
    from .build_pairwise import make_pairwise
except ImportError:
    from audit_datasets import audit
    from common import load_humarin_pairs, RAW_DIR, PROJECT_ROOT
    from build_triplet import sample_negatives, make_triplets
    from build_pairwise import make_pairwise


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("Expected a text string")
    return " ".join(unicodedata.normalize("NFC", text).split())


class Components:
    """Union source families whenever any normalized text is shared."""
    def __init__(self):
        self.parent = []
        self.owner = {}

    def add(self):
        i = len(self.parent)
        self.parent.append(i)
        return i

    def root(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def text(self, group, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        other = self.owner.setdefault(digest, group)
        a, b = self.root(group), self.root(other)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def read_padben(raw_dir, components):
    rows = []
    # Task3 and Task4 have two AI sides, regardless of original task labels.
    maps = {1: {0: 0, 1: 1}, 2: {0: 0, 1: 1}, 3: {0: 1, 1: 1},
            4: {0: 1, 1: 1}, 5: {0: 0, 1: 1}}
    for task in range(1, 6):
        path = Path(raw_dir) / f"padben_sentence_pair_task{task}.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in data:
            sentences, labels = row["sentence_pair"], row["label_pair"]
            if len(sentences) != 2 or len(labels) != 2 or sorted(labels) != [0, 1]:
                raise ValueError(f"Invalid pair schema in {path}")
            group = components.add()
            for position, (sentence, label) in enumerate(zip(sentences, labels)):
                sentence = normalize(sentence)
                if not sentence:
                    raise ValueError(f"Empty PADBen sentence in {path}")
                components.text(group, sentence)
                rows.append((group, sentence, maps[task][label], f"padben_task{task}",
                             task, label, row.get("idx"), position))
    return pd.DataFrame(rows, columns=["family_id", "sentence", "label", "source",
                                       "task", "original_task_label", "pair_id",
                                       "sentence_position"])


def prepare_sources(raw_dir, seed=42):
    pairs = load_humarin_pairs(str(Path(raw_dir) / "humarin_chatgpt_paraphrases.csv"))
    for col in ["original_text", "paraphrase_text"]:
        pairs[col] = pairs[col].map(normalize)
    pairs = pairs[(pairs.original_text != "") & (pairs.paraphrase_text != "") &
                  (pairs.original_text != pairs.paraphrase_text)].copy()
    components = Components()
    families = {g: components.add() for g in pairs.group_id.unique()}
    pairs["family_id"] = pairs.group_id.map(families)
    for group, original, paraphrase in pairs[["family_id", "original_text", "paraphrase_text"]].itertuples(index=False, name=None):
        components.text(group, original)
        components.text(group, paraphrase)
    padben = read_padben(raw_dir, components)
    roots = [components.root(i) for i in range(len(components.parent))]
    pairs["group_id"] = pairs.family_id.map(lambda i: roots[i])
    padben["group_id"] = padben.family_id.map(lambda i: roots[i])

    def split(group):
        value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
        return "train" if value < 0.7 else "val" if value < 0.85 else "test"

    assignments = {g: split(g) for g in set(roots)}
    pairs["split"] = pairs.group_id.map(assignments)
    padben["split"] = padben.group_id.map(assignments)
    pairs = pairs.drop_duplicates(["original_text", "paraphrase_text"]).reset_index(drop=True)
    human = pairs[["group_id", "original_text", "split"]].rename(columns={"original_text": "sentence"})
    ai = pairs[["group_id", "paraphrase_text", "split"]].rename(columns={"paraphrase_text": "sentence"})
    human = human.assign(label=0, source="humarin")
    ai = ai.assign(label=1, source="humarin")
    singles = pd.concat([human, ai, padben.loc[padben.task <= 3, ["group_id", "sentence", "split", "label", "source"]]], ignore_index=True)
    # Quarantine ambiguous labels instead of silently choosing one.
    label_evidence = pd.concat([singles[["sentence", "label"]], padben[["sentence", "label"]]], ignore_index=True)
    label_counts = label_evidence.groupby("sentence", sort=False).label.nunique()
    del label_evidence
    conflicts = label_counts.index[label_counts > 1]
    all_evidence = pd.concat([singles, padben[["group_id", "sentence", "split", "label", "source"]]], ignore_index=True)
    conflict_rows = all_evidence[all_evidence.sentence.isin(conflicts)].drop_duplicates()
    del all_evidence
    singles = singles[~singles.sentence.isin(conflicts)].drop_duplicates("sentence")
    padben = padben[~padben.sentence.isin(conflicts)].copy()
    manifest = pd.DataFrame({"family_id": np.arange(len(roots)), "group_id": roots})
    manifest["split"] = manifest.group_id.map(assignments)
    origin = {v: ("humarin", str(k)) for k, v in families.items()}
    for fid, source, pair_id in padben[["family_id", "source", "pair_id"]].drop_duplicates().itertuples(index=False, name=None):
        origin[fid] = (source, str(pair_id))
    manifest["source"] = manifest.family_id.map(lambda i: origin.get(i, ("padben", ""))[0])
    manifest["source_record_id"] = manifest.family_id.map(lambda i: origin.get(i, ("padben", ""))[1])
    stats = {"seed": seed, "target_step0": "AI-generated OR AI-paraphrased (AI-touched)",
             "split_policy": "70/15/15 hash of connected source components; approximate row ratios",
             "components": len(assignments), "normalized_texts": len(components.owner),
             "conflicting_sentence_labels_removed": len(conflicts),
             "largest_component_pairs": int(pairs.groupby("group_id").size().max())}
    return pairs, singles, padben, manifest, conflict_rows, stats


def build(raw_dir, output_dir, strategy="mixed", seed=42, chunk_size=512):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("Grouping source families and duplicate texts...", flush=True)
    pairs, singles, padben, manifest, conflicts, stats = prepare_sources(raw_dir, seed)
    manifest.to_csv(output / "source_manifest.csv", index=False)
    conflicts.to_csv(output / "label_conflicts.csv", index=False)
    vectorizer = None
    if strategy != "random":
        print("Fitting negative-mining TF-IDF on TRAIN only...", flush=True)
        train = pairs[pairs.split == "train"]
        vectorizer = TfidfVectorizer(max_features=100000, dtype=np.float32)
        vectorizer.fit(pd.concat([train.original_text, train.paraphrase_text]).drop_duplicates())
        del train
    stats["negative_strategy"] = strategy
    stats["chunk_size"] = chunk_size
    stats["splits"] = {}
    for split in ["train", "val", "test"]:
        print(f"Building {split} views...", flush=True)
        pool = pairs[pairs.split == split].reset_index(drop=True)
        selected, kinds = sample_negatives(pool, strategy, seed, chunk_size, vectorizer=vectorizer)
        triplets = make_triplets(pool, selected, kinds)
        pairwise = make_pairwise(triplets, seed)
        single = singles[singles.split == split].sample(frac=1, random_state=seed)
        triplets.to_csv(output / f"triplet_data_{split}.csv", index=False)
        pairwise.to_csv(output / f"pairwise_labeled_{split}.csv", index=False)
        single.to_csv(output / f"tang0_{split}.csv", index=False)
        stats["splits"][split] = {"triplets": len(triplets), "pairs": len(pairwise),
            "single": len(single), "groups": int(pool.group_id.nunique()),
            "negative_types": {str(k): int(v) for k, v in triplets.negative_type.value_counts().items()},
            "single_labels": {str(k): int(v) for k, v in single.label.value_counts().items()}}
        print(stats["splits"][split], flush=True)
        del pool, selected, kinds, triplets, pairwise, single
    for task in [4, 5]:
        heldout = padben[(padben.task == task) & (padben.split == "test")]
        heldout.to_csv(output / f"padben_task{task}_test.csv", index=False)
        stats[f"padben_task{task}_test_rows"] = len(heldout)
    (output / "data_report.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(RAW_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(PROJECT_ROOT) / "data" / "splits")
    parser.add_argument("--strategy", choices=["random", "hard", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw = args.raw_dir.resolve()
    if output == raw or output in raw.parents or raw in output.parents:
        raise ValueError("Output must be separate from raw input directory")
    if output.name in {"data", "src", ".git"} or output == Path(PROJECT_ROOT).resolve():
        raise ValueError("Choose a dedicated output directory, e.g. data/splits")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (".data-prep-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        build(args.raw_dir, staging, args.strategy, args.seed, args.chunk_size)
        audit(staging)
        archive = None
        if output.exists():
            archive = output.with_name(output.name + "_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            output.rename(archive)
            print(f"Previous outputs archived at {archive}", flush=True)
        try:
            staging.rename(output)
        except OSError:
            if archive is not None and not output.exists():
                archive.rename(output)
            raise
    except Exception:
        print(f"Build failed; partial diagnostics retained at {staging}", flush=True)
        raise
    print(f"Ready: {output}. See data_report.json before training.", flush=True)


if __name__ == "__main__":
    main()
