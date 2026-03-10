"""Синтетический японский smoke-набор."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import EMOTION_COLUMNS

BASE_CASES = (
    ("合格して本当にうれしい", "positive", {"joy", "trust"}),
    ("旅行の日が待ち遠しくて楽しみだ", "positive", {"joy", "anticipation"}),
    ("思いがけない贈り物に驚いて喜んだ", "positive", {"joy", "surprise"}),
    ("友人を信頼できて安心した", "positive", {"joy", "trust"}),
    ("試合に負けてとても悲しい", "negative", {"sadness"}),
    ("失礼な対応に腹が立ち嫌になった", "negative", {"anger", "disgust"}),
    ("突然の地震が怖くて驚いた", "negative", {"fear", "surprise"}),
    ("大切な物をなくして悲しく不安だ", "negative", {"sadness", "fear"}),
    ("今日は駅まで歩いた", "neutral", set()),
    ("明日の会議を予定している", "neutral", {"anticipation"}),
    ("担当者から通常の連絡を受けた", "neutral", set()),
    ("予定が変更されたと聞いた", "neutral", {"surprise"}),
)


def make_smoke_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    text_number = 1
    for variant in range(1, 5):
        for text, polarity, emotions in BASE_CASES:
            row: dict[str, object] = {
                "text_id": f"jp-{text_number:03d}",
                "text": f"{text}。例{variant}",
                "polarity": polarity,
            }
            for column in EMOTION_COLUMNS:
                row[column] = int(column.removeprefix("emotion_") in emotions)
            rows.append(row)
            text_number += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_smoke_data().to_csv(destination, index=False)


if __name__ == "__main__":
    main()
