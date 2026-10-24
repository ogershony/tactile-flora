"""Write the zero-shot camera/resolution study jobs (released FLORA, TRAIN subset only).

Phase 1 configs: legacy input (scene camera 22641005 in a black-padded 96 px cell) and every
candidate camera alone at the released cell size. Phase 2 (--multi cam1,cam2,...): the given
cameras as 2-, 3- and 4-view collages in the given order. Task text is the same for every config.
"""

import json
import sys
from pathlib import Path

split = json.loads(Path(sys.argv[1]).read_text())
frames = Path(sys.argv[2])
out = Path(sys.argv[3])
phase = sys.argv[4]
cams = sys.argv[5].split(",")
eps = split["episodes"]
jobs = []
for ep in split["camera_study_train_subset"]:
    d = frames / ep
    if not (d / "done.json").is_file():
        continue
    meta = json.loads((d / "video.json").read_text())
    if any(meta.get(c, {}).get("missing", c not in meta) for c in cams):
        continue  # a candidate camera is absent upstream for this trial
    info = eps[ep]
    obj = info["object_id"].replace("_", " ")
    base = {
        "task": f"Grasp the {obj}, lift it off the table and hold it in the air for 10 seconds.",
        "frames_dir": str(d),
        "label": 5 if info["success"] else 1,
        "group": info["object_id"],
    }
    if phase == "1":
        jobs.append({**base, "id": f"{ep}|cell96|22641005", "cams": ["22641005"], "cell96": True})
        for c in cams:
            jobs.append({**base, "id": f"{ep}|native|{c}", "cams": [c]})
    else:
        for k in range(2, len(cams) + 1):
            jobs.append({**base, "id": f"{ep}|native|{'+'.join(cams[:k])}", "cams": cams[:k]})
out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
print(len(jobs), "jobs ->", out)
