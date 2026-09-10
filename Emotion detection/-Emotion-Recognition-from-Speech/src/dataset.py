"""Dataset indexing and the actor-disjoint (speaker-independent) split.

RAVDESS encodes every label in the filename, e.g.::

    03-01-06-01-02-01-12.wav
    |  |  |  |  |  |  +-- actor      (01..24, odd = male, even = female)
    |  |  |  |  |  +----- repetition (01, 02)
    |  |  |  |  +-------- statement  (01, 02)
    |  |  |  +----------- intensity  (01 normal, 02 strong)
    |  |  +-------------- emotion    (01..08)
    |  +----------------- vocal channel (01 speech, 02 song)
    +-------------------- modality   (03 = audio-only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def parse_filename(path: Path) -> dict | None:
    """Decode a RAVDESS filename into its label fields. Returns None if malformed."""
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if emotion not in C.EMOTION_MAP:
        return None
    actor_id = int(actor)
    return {
        "path": str(path),
        "filename": path.name,
        "modality": modality,
        "vocal_channel": channel,
        "emotion_code": emotion,
        "emotion": C.EMOTION_MAP[emotion],
        "intensity": C.INTENSITY_MAP.get(intensity, "unknown"),
        "statement": C.STATEMENT_MAP.get(statement, "unknown"),
        "repetition": int(repetition),
        "actor": actor_id,
        "gender": "male" if actor_id % 2 == 1 else "female",
    }


def build_manifest(root: Path = C.RAVDESS_DIR) -> pd.DataFrame:
    """Walk the dataset directory and build a dataframe of every audio file."""
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {root}")
    rows, bad = [], []
    for w in wavs:
        parsed = parse_filename(w)
        (rows if parsed else bad).append(parsed if parsed else w.name)
    if bad:
        print(f"[warn] {len(bad)} files had unparsable names, e.g. {bad[:3]}")
    df = pd.DataFrame(rows)
    # Speech-only guard: this project deliberately excludes the song subset.
    df = df[df["vocal_channel"] == "01"].reset_index(drop=True)
    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a train/val/test column using an ACTOR-DISJOINT partition.

    A random file-level split would place several recordings of the *same*
    actor in both train and test.  The network could then latch onto speaker
    identity instead of emotion, inflating the reported score.  Splitting by
    actor guarantees every test utterance comes from a voice the model has
    never heard.
    """
    def which(actor: int) -> str:
        if actor in C.TEST_ACTORS:
            return "test"
        if actor in C.VAL_ACTORS:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df["actor"].map(which)

    # Hard assertion: no actor may appear in more than one split.
    groups = {s: set(g["actor"]) for s, g in df.groupby("split")}
    assert not (groups["train"] & groups["val"]), "actor leak train/val"
    assert not (groups["train"] & groups["test"]), "actor leak train/test"
    assert not (groups["val"] & groups["test"]), "actor leak val/test"
    return df


def get_manifest() -> pd.DataFrame:
    """Manifest with split assigned (cached to data/processed/manifest.csv)."""
    cache = C.DATA_PROCESSED / "manifest.csv"
    if cache.exists():
        return pd.read_csv(cache)
    df = assign_split(build_manifest())
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    d = assign_split(build_manifest())
    d.to_csv(C.DATA_PROCESSED / "manifest.csv", index=False)
    print(f"files: {len(d)}")
    print(d.groupby("split")["actor"].nunique().rename("actors"))
    print(d["split"].value_counts())
