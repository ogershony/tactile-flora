"""Print per-clip scores of every eval cell as compact text (to copy results off the runtime).

For each benchmark: one header line ``B <bench> n=<n> sha=<sha256 of the sorted window ids, 12>``,
the truth digits in window-id order (``T`` line), then one ``C <checkpoint>|<mode>`` line per cell
with round(100 * E[s]) and the hard score per window (``<es100>/<score>``, ``-`` when the cell has
no row for that window). Windows are sorted by window_id, the same order on any machine that
built the same manifests. Lines wrap at --width characters with a trailing backslash. --pack writes each window as two
characters of ALPHABET instead: v = (round(100 E[s]) - 100) * 5 + (score - 1), 0..2004, as
ALPHABET[v // 64] + ALPHABET[v % 64]; "~~" = no row, "!!" = NaN (decode with unpack_value).
Usage: python scripts/v4/export_scores.py data/eval [bench ...] [--width 79] [--pack]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def pack_value(es: float, score: float) -> str:
    """Two characters for one window (see the module docstring)."""
    if pd.isna(es) or pd.isna(score):
        return "!!"
    v = (int(round(100 * float(es))) - 100) * 5 + int(score) - 1
    return ALPHABET[v // 64] + ALPHABET[v % 64]


def unpack_value(two: str) -> tuple[float, int] | None:
    """Inverse of pack_value: (E[s] to 0.01, hard score), or None for a missing or NaN window."""
    if two in ("~~", "!!"):
        return None
    v = ALPHABET.index(two[0]) * 64 + ALPHABET.index(two[1])
    return (100 + v // 5) / 100, v % 5 + 1


def wrap(line: str, width: int) -> list[str]:
    """Split ``line`` into chunks of width - 1 characters, each but the last ending in '\\'."""
    step = width - 1
    parts = [line[i : i + step] for i in range(0, len(line), step)] or [""]
    return [p + "\\" for p in parts[:-1]] + [parts[-1]]


def main() -> None:
    """Print the export."""
    pack = "--pack" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 79
    if "--width" in sys.argv:
        args.remove(str(width))
    root, want = Path(args[0]), set(args[1:])
    cells: dict[str, dict[str, pd.DataFrame]] = {}
    for f in sorted(root.glob("*/*/*/predictions.parquet")):
        ck, bench, mode = f.parts[-4], f.parts[-3], f.parts[-2]
        if want and bench not in want:
            continue
        p = pd.read_parquet(f, columns=["window_id", "expected_score", "score", "truth_target"])
        cells.setdefault(bench, {})[f"{ck}|{mode}"] = p.set_index("window_id")
    for bench in sorted(cells):
        ids = sorted(set().union(*(set(c.index) for c in cells[bench].values())))
        sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:12]
        print(f"B {bench} n={len(ids)} sha={sha}")
        truth = pd.Series("-", index=ids)
        for c in cells[bench].values():
            t = c.truth_target.str.extract(r"Score:\s*(\d)")[0]
            truth.loc[t.index] = t.fillna("-")
        for ln in wrap("T " + "".join(truth.loc[ids]), width):
            print(ln)
        for key in sorted(cells[bench]):
            c = cells[bench][key]
            vals = []
            for w in ids:
                if w not in c.index:
                    vals.append("~~" if pack else "-")
                    continue
                es, sc = c.at[w, "expected_score"], c.at[w, "score"]
                if pack:
                    vals.append(pack_value(es, sc))
                    continue
                es_s = "n" if pd.isna(es) else str(int(round(100 * float(es))))
                sc_s = "n" if pd.isna(sc) else str(int(sc))
                vals.append(f"{es_s}/{sc_s}")
            text = f"C {key} " + ("".join(vals) if pack else " ".join(vals))
            for ln in wrap(text, width):
                print(ln)


if __name__ == "__main__":
    main()
