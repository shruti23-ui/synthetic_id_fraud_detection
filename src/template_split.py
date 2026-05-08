"""
src/template_split.py
---------------------
Canonical template-aware 60/20/20 splitter for the local fake/real ID
dataset.

Each fake document is derived from a *source template* (the real ID it
was forged from). Splitting images uniformly at random allows the same
template to land in train as a real and in test as one of its derived
fakes, giving the model a route to score by template-identity rather
than by detecting forgery cues. The data-leakage audit
(`research/01_data_leakage_audit.py`) found 65 % source-template
overlap and a 99 % near-duplicate rate under the naive image-level
split, motivating this template-aware splitter.

Algorithm:
  1. Group records by their source template ID.
  2. Greedy-assign each template (in deterministic shuffled order) to
     whichever split currently has the largest deficit relative to its
     target image-count. This balances split sizes while guaranteeing
     ZERO template overlap between any two splits.

Used by:
  * src/supervised_finetune.py            (Stage 8 of main.py)
  * src/train_two_stream.py               (Stage 9 of main.py)
  * research/02_template_split_retrain.py (standalone audit-recovery run)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def template_id(record: dict) -> str:
    """Return the underlying template ID (e.g. 'alb_id_00') for any record.

    For real images the template ID is the file stem itself.
    For fake images it is the `src` field of the per-image VIA-format
    annotation JSON (e.g. ``alb_id_00.jpg`` -> ``alb_id_00``); if that
    field is missing, fall back to the heuristic of stripping the
    ``_fake_*`` suffix from the file stem.
    """
    label = record["label"]
    stem = record["path"].stem

    if label == 0:
        return stem

    annot = record.get("annotation") or {}
    src = annot.get("src", "")
    if src:
        return Path(src).stem
    parts = stem.split("_fake_")
    return parts[0] if parts else stem


def template_aware_split(
    records: list[dict],
    cfg: dict,
) -> tuple[list[int], list[int], list[int]]:
    """Greedy template-aware split.

    Args:
        records: List of record dicts (each with ``label``, ``path``, ``annotation``).
        cfg: Dict with keys ``seed`` (int), ``val_size`` (float), ``test_size`` (float).

    Returns:
        Three sorted lists of indices into ``records`` for train / val / test.

    Guarantees (asserted at exit):
        * No template appears in more than one split.
        * Class distribution is approximately preserved across splits because
          all of a template's derived fakes (and the corresponding real)
          go into the same split together.
    """
    tpl_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        tpl_to_idx[template_id(rec)].append(i)

    template_ids = sorted(tpl_to_idx.keys())
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(template_ids)

    n_total      = sum(len(v) for v in tpl_to_idx.values())
    target_test  = int(round(n_total * cfg["test_size"]))
    target_val   = int(round(n_total * cfg["val_size"]))
    target_train = n_total - target_test - target_val

    splits  = {"train": [], "val": [], "test": []}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    for tid in template_ids:
        deficits = {k: targets[k] - len(splits[k]) for k in splits}
        chosen = max(deficits, key=lambda k: deficits[k])
        splits[chosen].extend(tpl_to_idx[tid])

    train_tpl = {template_id(records[i]) for i in splits["train"]}
    val_tpl   = {template_id(records[i]) for i in splits["val"]}
    test_tpl  = {template_id(records[i]) for i in splits["test"]}
    assert not (train_tpl & val_tpl),  "template leak between train and val"
    assert not (train_tpl & test_tpl), "template leak between train and test"
    assert not (val_tpl   & test_tpl), "template leak between val and test"

    return sorted(splits["train"]), sorted(splits["val"]), sorted(splits["test"])
