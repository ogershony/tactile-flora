"""Decode export_scores.py --pack output into one long CSV (bench, row, truth, checkpoint, mode, es, score).

Rows follow the sorted window ids of each benchmark's manifest (the B line's sha checks them). Prints
the failure-positive AUROC of every cell as a check against results.json.
Usage: python scripts/v4/decode_export.py docs/v4/results/export_packed.txt out.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def unpack(two: str) -> tuple[float, int] | None:
    if two in ("~~", "!!"):
        return None
    v = ALPHABET.index(two[0]) * 64 + ALPHABET.index(two[1])
    return (100 + v // 5) / 100, v % 5 + 1


def auroc(y: list[int], risk: list[float]) -> float:
    pairs = sorted(zip(risk, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    s = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    lines, cur = [], ""
    for raw in Path(sys.argv[1]).read_text().splitlines():
        if raw.endswith("\\"):
            cur += raw[:-1]
        else:
            lines.append(cur + raw)
            cur = ""
    rows, bench, truth = [], None, []
    for ln in lines:
        if ln.startswith("B "):
            bench = ln.split()[1]
        elif ln.startswith("T "):
            truth = [None if c == "-" else int(c) for c in ln[2:]]
        elif ln.startswith("C "):
            _, key, packed = ln.split(" ", 2)
            ck, mode = key.split("|")
            vals = [unpack(packed[i : i + 2]) for i in range(0, len(packed), 2)]
            assert len(vals) == len(truth), (bench, key, len(vals), len(truth))
            y, risk = [], []
            for r, (t, v) in enumerate(zip(truth, vals, strict=True)):
                if v is None or t is None:
                    continue
                rows.append((bench, r, t, ck, mode, v[0], v[1]))
                y.append(int(t <= 2))
                risk.append(-v[0])
            print(f"{bench:20s} {ck:16s} {mode:9s} n {len(y)} AUROC {auroc(y, risk):.3f}")
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "row", "truth", "checkpoint", "mode", "es", "score"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
