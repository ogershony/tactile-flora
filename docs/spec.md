
# Tactile FLORA: Technical Specification

2026-10-21 · @Oren Gershony

## Purpose and scope

This document specifies tactile-flora, a Python repository that adds a fingertip-force channel to NVIDIA's FLORA-Rewarder (Qwen3.5-27B) by rendering force as text in the prompt, fine-tunes it with LoRA on public force datasets, and evaluates it on labeled tactile sets and on a Baxter robot. It is written for an implementing agent; the companion planning doc holds the schedule and the rationale, this one holds what to build.
In scope: dataset adapters, force resampling and text rendering, label mapping, data verification and manifest, FLORA prompt construction, LoRA SFT, an evaluation harness with ablation modes, and the Baxter capture stack (firmware, ROS node, dataset writer). Out of scope: the captioner and segmenter, a learned force encoder, simulation generation, and any VLA training; the design must not block them later.
Conventions the implementer follows:
- Python 3.11, uv for environments, ruff and pyright clean, type hints everywhere, pydantic models for every schema that crosses a module boundary.
- One responsibility per module, one public entry point per module, docstring at the top of every module stating what it owns and what it must never do.
- Every stage reads from and writes to files on disk with a documented schema (parquet or JSON); no stage calls another stage's internals. This is what makes the repo readable and lets stages run on different machines (MI210s, cloud GPU, Baxter workstation).
- Config is YAML under configs/, loaded into pydantic models; no magic constants in code. Every CLI command takes --config.
- Nothing in training or eval reads raw datasets directly; both read only the accepted manifest.
- Keep the README current: every top-level command appears there with a one-line example.

## Background and hypothesis

FLORA (NVIDIA GEAR, October 2026, paper, weights in the nvidia/flora Hugging Face collection) is a data engine of three fine-tuned Qwen3.5 VLMs: a 9B segmenter, a 27B captioner and a 27B reward model. FLORA-Rewarder takes a multi-camera video mosaic plus a task instruction and emits either an outcome score (1 to 5) or a dense execution-quality sequence (one digit in 1 failure / 2 suboptimal / 3 normal per 0.5 s group over a 16 s window, with rationales for non-normal runs). It is vision-only.
Hypothesis under test: fingertip force reveals contact failures that RGB misses or sees late (grasp acquired or not, slip onset, excessive force on compliant objects, premature release, off-axis grasp). Injecting a normalized force stream as text into the rewarder's prompt and fine-tuning with LoRA should raise failure recall and improve outcome scores on contact failures, without degrading vision-only reward judgment.
Decisions already taken in planning, which the implementation must respect:
- Rewarder only. The captioner is excluded because force reaches at most the 5% "detail" weight of its metric and captions must not encode outcomes; the segmenter is a conditional follow-on.
- Force enters as text, not a learned encoder, for this iteration; the resampler keeps the raw 100 Hz slice per group so an encoder can be added later without touching data.
- Single-arm evaluation. No public dataset combines bimanual manipulation, force and quality labels; the model and the force block already support two arms, so bimanual is a data question, not a code change.
- Every claim is backed by a matched no-force control trained on identical data and by a force-shuffle ablation; a gain that survives shuffling is not a force gain.

## System overview

Public datasets and Baxter recordings enter through adapters that put every stream on one episode clock; the force channel, labels and verification run over those episodes and produce an accepted manifest; prompts, training and evaluation read the manifest and nothing else. The Baxter adapter sits beside the public ones so robot data is treated identically.

## FLORA reference facts

Numbers the implementation reproduces, taken from the paper; the implementer should re-read Appendix G of the paper for the prompt text and keep this table as the checklist.
| Item | Value |
| Reward model base | Qwen3.5-27B; SFT only in the released checkpoint; language model, visual merger and visual encoder were all tuned (encoder at lower LR) |
| Dense window | 64 frames at 4 FPS spanning [alpha minus 2, alpha plus 14) s; anchors alpha_k = 10k; final window at t_last minus 9.75; 32 groups of 2 frames; groups 25 to 32 dropped at stitching for non-final windows |
| Dense output | Worst: <1-3> then 32 digits then Rationales: lines <first>-<last> <text> for each contiguous non-normal run; greedy, regex-constrained, up to 640 new tokens; Worst must equal the minimum digit or the answer is rejected and replaced by all-normal |
| Outcome input | m = min(64, max(2, floor(10 N / fps))) frames rounded down to even, indices round(i (N-1)/(m-1)); score-only mode 12 tokens, rationale mode 160 |
| Outcome continuous reward | E[s] from the top-20 alternatives at the score token, keep digits 1 to 5, renormalize; y = (E[s] minus 1)/4 |
| Mosaic | up to 4 synchronized views per sampled time; 1x1, 1x2, 1x3 or 2x2 by camera roles; scene view upper left in 2x2; cells multiples of 32 px, about 105 px per view; 64-frame clip capped at 14,336 visual tokens |
| Timestamps | source frame indices and fps passed to the Qwen3.5 video processor so timestamp tokens are absolute episode seconds |
| Task text | global instruction only, outcome words removed (failure, success, retry, recovery) |
| Released public metrics | RoboRewardBench MAE 40.0 (x100); RobometerBench OOD Kendall 72.0, ID VOC 98.9, OOD VOC 99.6 |
| Precision note | Gated DeltaNet parameters kept in fp32 at inference; bf16 elsewhere |

## Repository layout

One package, tactile_flora, with a subpackage per stage. Stages communicate only through files under data/ whose schemas live in tactile_flora/schema.py.
```
tactile-flora/
  README.md                 quickstart, one line per command
  pyproject.toml            uv-managed; extras: [train], [baxter], [dev]
  configs/
    sources/                one YAML per dataset adapter (paths, splits, normalization)
    force.yaml              resampling rates, schema, text format
    train/                  force.yaml and control.yaml (identical except force block)
    eval/                   benchmark definitions and ablation grid
    baxter.yaml             sensor calibration, serial port, topics
  tactile_flora/
    schema.py               pydantic models: Episode, ForceStream, Window, Label, ManifestRow
    sources/                one adapter per dataset; all implement SourceAdapter
      base.py               SourceAdapter interface + registry
      phantom.py sharpadex.py rh20t.py reassemble.py airoa.py
      hrdexdb.py softvtbench.py baxter.py      (eval-only adapters)
    force/
      resample.py           raw stream to frame-aligned features
      normalize.py          per-source constants, fit and apply
      render.py             features to text lines
    labels/
      mapping.py            per-source label to FLORA label
      events.py             programmatic force events (slip, crush, regrasp, contact loss)
      pseudo.py             vision-only FLORA pseudo-labeling
      windows.py            FLORA window geometry, dense label stitching
    verify/
      checks.py             one function per check, returns CheckResult
      hashing.py            perceptual frame hashes, overlap detection
      report.py             per-source markdown report
      manifest.py           write and read the accepted manifest
    prompts/
      flora.py              dense and outcome prompt builders, exact FLORA grammar
      parse.py              output parsers (Worst/Quality/Rationales; Score)
    train/
      dataset.py            manifest to tokenized samples
      sft.py                LoRA SFT loop (HF + PEFT + FSDP)
    eval/
      harness.py            checkpoint x benchmark x force_mode to predictions.parquet
      metrics.py            MAE, macro F1, balanced acc, bad-action F1, failure recall, AUROC
      ablations.py          force modes: real, omitted, zero, shuffled, video_blank
    baxter/
      firmware/teensy_fsr/  Arduino sketch, 4 channels at 200 Hz
      ros/                  fsr_node.py (publisher), recorder.py (episode capture)
      calibrate.py          weights to per-pad curve
      to_lerobot.py         episodes to LeRobot v3 with force columns
  scripts/
    tf                      single CLI entry: tf verify | tf manifest | tf pseudo | tf train | tf eval | tf baxter ...
  tests/
    unit/                   one test file per module
    fixtures/               tiny synthetic episodes (10 frames, synthetic force)
  data/                     gitignored; raw/, verified/, manifest/, runs/, eval/
```
Rule of thumb for the implementer: if a function needs to import from two sibling stages, it belongs in schema.py or it is a sign the stages are coupled wrongly.

## Core data model

Five pydantic models in schema.py carry everything between stages. Timestamps are float seconds on the episode clock, zero at the first video frame; every stream is converted to that clock by its adapter and never resampled again except by force/resample.py.
```
Episode
  source: str              # "phantom" | "sharpadex" | "rh20t" | ...
  episode_id: str          # source-native id, unique within source
  session: str             # collection session / season / scene; drives splits
  task: str                # instruction text handed to FLORA, outcome words stripped
  duration_s: float
  cameras: dict[str, VideoRef]   # role -> path + fps + frame count; roles: scene, wrist_left, wrist_right, ...
  force: ForceStream
  gripper: GripperStream | None  # position (0 open .. 1 closed) per arm at native rate
  labels: list[Label]
  license: str
  meta: dict               # anything source-specific, never read downstream

ForceStream
  channel: "pad" | "wrist"
  arms: dict["L"|"R", ArmForce]
  rate_hz: float
  t: float[N]              # episode clock
ArmForce (pad)   : f_a float[N], f_b float[N]          # two pad readings, raw units
ArmForce (wrist) : fx fy fz tx ty tz float[N]          # raw units

Label
  kind: "outcome" | "dense" | "event"
  origin: "human" | "programmatic" | "pseudo" | "source"
  outcome: int 1..5 | None
  spans: list[(t0, t1, quality 1|2|3, rationale)]      # dense and event

Window                     # a training or eval sample
  episode_ref, prompt_kind: "dense" | "outcome"
  anchor_s: float          # dense: alpha_k; outcome: 0
  frame_times: float[64]
  force_lines: list[str]   # rendered text, empty for control
  target: str              # exact FLORA answer string
  split: "train" | "val" | "test" | "benchmark"

ManifestRow               # one row per accepted Window, data/manifest/<name>.parquet
  window_id, source, episode_id, session, split, prompt_kind, anchor_s
  frame_hashes: list[str]  # first, middle, last frame phash
  norm_version: str        # hash of the normalization constants used
  label_origin, license, report_version
```
Two invariants the implementer enforces with assertions and unit tests: a Window is immutable once written, and a ManifestRow's split never changes after the held-out id file is frozen (data/manifest/holdout_ids.json).

## Source adapters

Every dataset is wrapped by one class implementing SourceAdapter; nothing else in the repo knows how a dataset is laid out on disk. Adapters are lazy (they yield Episode objects one at a time) and pure (they never write).
```
class SourceAdapter(Protocol):
    name: str
    def discover(self, root: Path) -> list[str]          # episode ids
    def load(self, episode_id: str) -> Episode            # full episode on the episode clock
    def splits(self) -> dict[str, list[str]]              # session-level split assignment
    def native_labels(self, episode_id: str) -> list[Label]   # origin="source"
```
| Adapter | Reads | Force mapping | Gripper | Native labels | Split key |
| phantom | LeRobot episodes + manifests/all.jsonl | DM-Tac force and contact area per pad, channel pad | Robotiq 2F-85 position | success flag; *_fail task folders | manifest split; failure demos train-only |
| sharpadex | LeRobot v3 parquet + videos, meta/subtasks.parquet | observation.tactile 60-D, take right thumb and index fz as f_a, f_b, channel pad | hand joint state, closure proxy from index MCP | none (subtask index kept in meta) | season directory |
| rh20t | community LeRobot v3 ports (robot-lev/rh20t_cfg*) | observation.robot_ft 6-D, channel wrist | observation.state[-1] | meta.rating 0 to 9 | scene (also carries license) |
| reassemble | HDF5 per demonstration | wrist F/T 6-D, channel wrist | gripper state | per-action success, anomaly spans, segment boundaries | collection session |
| airoa | LeRobot v2.1 | wrist F/T 6-D, channel wrist | hand_motor | episode success flag with cause; per-PA O/X | robot id + date |
| hrdexdb (eval only) | per-trial archives | fingertip tactile per hand; adapter maps to pad using the two most-loaded fingers | hand state | binary grasp success | robot hand type |
| softvtbench (eval only) | HDF5 + MP4 | FEM contact state to scalar force proxy per finger, channel pad; deformation ratio kept in meta | continuous gripper action | task success, DSR, peak normalized deformation | task id |
| baxter (eval only) | LeRobot v3 written by baxter/to_lerobot.py | four pad channels, channel pad | Baxter gripper position | human labels from labels.json | recording day |
Adapter rules:
- Convert every stream to the episode clock in load(); store the offset used in meta["clock_offset_s"].
- Strip outcome words from task (fail, success, retry, recover, crush) with the shared regex in sources/base.py, the same way the paper does for its dense prompt.
- Never filter episodes inside an adapter; discover() returns everything and verify/ decides.
- Each adapter ships a tests/fixtures/<name>/ directory with one tiny synthetic episode so unit tests run without the real datasets.
- Register with @register_source("name"); the CLI resolves adapters by name from configs/sources/<name>.yaml.

## Data sources, caps and exclusions

The training manifest draws from five public sources with per-source caps (configs/sources/<name>.yaml: max_windows, max_episodes), about 1,900 samples total. Pseudo-labeled replay stays under a third of the mix; force-specific supervision comes only from PHANTOM, REASSEMBLE, AIRoA and programmatic events.
| Source | Where | Force | Cap | Prompt | License |
| PHANTOM | huggingface armteam/phantom-episodes | fingertip DM-Tac (manifest force, contact area) | ~300 dense windows, ~100 outcome episodes | dense + outcome | check card |
| SharpaDex v1.0 | huggingface Sharpa-Robotics/SharpaDex-v1.0 (3 to 4 seasons) | fingertip 6-axis F/T, 10 fingers | ~600 dense windows (pseudo + events) | dense | CC-BY-4.0 |
| RH20T | huggingface robot-lev/rh20t_cfg* LeRobot v3 ports | wrist 6-axis F/T | ~300 outcome episodes | outcome | CC BY-SA 4.0 scenes 1 to 5, CC BY-NC 4.0 scenes 6 to 10, flagged per row |
| REASSEMBLE | TU Wien research data | wrist 6-axis F/T | ~300 outcome episodes, ~150 dense windows | outcome + dense | open |
| AIRoA MoMa | huggingface airoa-org/airoa-moma | wrist 6-axis F/T, Toyota HSR | ~200 outcome episodes, ~100 dense windows | outcome + dense | check card |
Eval-only sources (never loaded by training or pseudo-labeling): PHANTOM val split, REASSEMBLE held-out sessions, HRDexDB, SoftVTBench (Apache-2.0, 4,000 demos, FEM deformation ground truth), RoboRewardBench stratified 500-episode subsample (vision-only regression check), and the Baxter set.
Excluded and why, so nobody re-adds them by accident: Intern pour/screw/shiguan sets (no license declared); TacClaw-Infinity (CC BY-NC-SA, no quality labels); FTP-1-Dataset, T-Rex, Xense sets (no quality labels; reserved for encoder pretraining later); RoboMIND and AgiBot World (bimanual, reserved for the bimanual follow-on); RobometerBench (dropped in favor of tactile benchmarks).
Split rules: by episode and, where the source has them, by session (PHANTOM manifest split, SharpaDex season, REASSEMBLE and AIRoA collection session, RH20T scene). Never by frame or window.

## Force channel

Three pure functions in force/, applied in order: resample to the frames FLORA will see, normalize with per-source constants, render to one text line per frame. The same three functions serve training, eval and the Baxter node; there is exactly one implementation of each.
force/resample.py takes a ForceStream and a list of frame times and returns one feature vector per frame per arm. For each frame time t it uses the raw samples in [t minus 0.125 s, t plus 0.125 s] (half the 4 Hz frame period on each side):
```
F_t = \operatorname{median}(f^a + f^b),\quad \Delta_t = \operatorname{median}(f^a - f^b),\quad \dot F_t = \frac{F_{t} - F_{t-0.25}}{0.25},\quad c_t = \mathbb{1}[F_t > \tau]
```
For wrist channels F is the median wrench magnitude and Delta is omitted. The gripper position g_t is sampled at the nearest timestamp. The function also returns the raw 100 Hz slice per 0.5 s group so a later learned encoder can reuse the same geometry without changing the resampler.
force/normalize.py fits constants per source and stores them in data/verified/<source>/norm.json with a content hash (norm_version):
- baseline: median F over frames where the gripper is open and no contact is expected (first 0.5 s of every episode).
- grasp: median F over frames where the gripper is closed on an object (g above 0.6 and F above baseline plus 3 sigma of the free-space noise).
- tau: baseline plus 3 sigma.
Applied as F' = clip((F minus baseline) / (grasp minus baseline), 0, 3); Delta and dF are divided by the same scale. Eval-only sources fit their own constants on their own data; the fitter never sees training data for them. Baxter fits constants per calibration file, not per session.
force/render.py produces the text block. Exact format, one line per sampled frame, fixed-width numbers with two decimals, arms in L then R order, omitted arm written as none:
```
t=3.50 pad L F=0.62 D=-0.08 dF=0.41 g=0.31 c=1 | pad R F=0.00 D=0.00 dF=0.00 g=1.00 c=0
t=3.75 wrist L F=0.31 dF=-0.12 g=0.52 c=1 | R none
```
The block is preceded by the header line Force (normalized: 0 free space, 1 median grasp, max 3): and is inserted after the video token and before the task line in both prompts. The control condition renders an empty block and omits the header. Token budget check in render.py asserts under 1,500 tokens per 64-frame window using the Qwen tokenizer.
Ablation modes live in eval/ablations.py and transform the resampled features before rendering: real (identity), omitted (no block, no header), zero (all values 0.00, c=0), shuffled (features from a different episode of the same source, seeded), video_blank (features unchanged, frames replaced by mid-gray).

## Labels

Labels come from four origins and are merged into FLORA's two targets (an outcome score 1 to 5; a dense sequence of 32 quality digits per 16 s window with rationales for non-normal runs). Each origin is its own module so a reviewer can see exactly where every digit came from.
labels/mapping.py converts native source labels to Label objects:
| Source | Outcome | Dense |
| PHANTOM | success=true to 5; *_fail to 1 | failure span from the first frame where normalized F exceeds that object's success-set 99th percentile until episode end, rationale excessive force on the <object> |
| RH20T | rating 0 to 1; rating 1 to 2 (robot fault absent, goal missed; use 3 if the object was moved toward the goal, decided by a single rule in the mapping with a unit test); 2 to 9 to 5 | none |
| REASSEMBLE | per-action success to 5 or 1 at the episode level (any failed action to 1) | anomaly annotations to failure spans, rationale from the annotation text |
| AIRoA MoMa | episode flag to 5 or 1 | failed primitive-action segments to failure spans, rationale from the documented cause |
| SharpaDex | none | none (pseudo + events only) |
labels/events.py derives programmatic spans from normalized force and gripper position. Each detector is a small pure function with a docstring giving the physical rationale and the thresholds live in configs/force.yaml:
- grasp_acquired: g rises above 0.6 and F' rises above 0.5 within 0.5 s. Not a label, used by other detectors and by verification.
- slip: dF' below minus 2.0 for at least 0.1 s while g is constant within 0.02 and the object was held. Quality 2 (suboptimal); rationale the object slips in the <arm> gripper.
- contact_loss: F' drops below tau for over 0.3 s while g stays closed. Quality 1 if the task was not complete; rationale the object drops from the <arm> gripper.
- regrasp: two grasp_acquired events within 3 s with a contact_loss between them. Quality 2; rationale re-grasps the object.
- crush: F' above 2.0 (relative to the median grasp) while g is still closing. Quality 1; rationale excessive force on the object.
- off_axis: absolute Delta' above 0.6 during a hold. Quality 2; rationale off-center grasp.
Precedence when spans overlap: human > source > programmatic > pseudo; within a frame, the more severe quality wins (1 over 2 over 3), as in the paper.
labels/pseudo.py runs the released vision-only FLORA-Rewarder (force omitted) over SharpaDex windows and stores the parsed digits and rationales with origin="pseudo". It refuses to run on any source marked eval_only and asserts the output is not already in holdout_ids.json.
labels/windows.py implements the paper's geometry exactly: anchors alpha_k = 10k while alpha_k < t_last minus 9.75, then a final window at that bound; 64 positions at alpha_k plus (i minus 9)/4 s; group g covers positions 2g minus 1 and 2g; each group takes its more severe frame label; padded groups take the boundary frame's label; stitching drops groups 25 to 32 of every non-final window. Outcome samples take m = min(64, max(2, floor(10 N / fps))) frames rounded down to even, indices round(i (N minus 1)/(m minus 1)).
The dense target string is produced by one function to_flora_dense_target(spans, window) -> str and the outcome target by to_flora_outcome_target(label, with_rationale: bool) -> str. Both are the inverses of the parsers in prompts/parse.py, and a round-trip unit test guards that.

## Verification pipeline and manifest

tf verify --source <name> runs every check in verify/checks.py over every episode the adapter discovers, writes data/verified/<source>/report.md and checks.parquet (one row per episode per check), and tf manifest turns passing episodes into Window rows in data/manifest/<name>.parquet. Training and eval read only the manifest.
Each check is a function check_<name>(episode: Episode, ctx: VerifyContext) -> CheckResult where CheckResult has passed: bool, value: float | None, action: "keep" | "drop_episode" | "drop_channel" | "shift", detail: str. Checks never mutate the episode; shift records a clock correction that the manifest stores and the resampler applies.
| Check | Rule | Action on fail |
| integrity | videos decode, frame count matches rows, timestamps monotonic, force present at declared rate | drop_episode |
| sensor_health | per channel: variance above noise floor, saturation fraction under 1%, per-session baseline within 2 sigma | drop_channel or drop_episode if both pads fail |
| sync | cross-correlation lag between F and gripper closure, median under 100 ms, max 250 ms | shift (store lag) or drop_episode |
| normalization | after applying constants: median grasp near 1.0, free space near 0.0, 99th percentile under 3 | fail the source, fix constants |
| label_sanity | class balance within 2x of the expected per-source rate (configured) | fail the source |
| spot_check | writes 20 random windows per source to data/verified/<source>/spot/ as video plus label strip for a human; the CLI records the reviewer's pass count | fail if under 17 of 20 |
| prompt_diff | force and control prompts for the same window are byte-identical outside the force block; block under 1,500 tokens | fail the source |
| holdout | no episode id in holdout_ids.json; no pseudo-labeled window in any eval split | drop_episode |
| overlap | perceptual hash (pHash, 64 bit) of first, middle, last frames vs every benchmark episode; Hamming distance under 6 on all three is a match | drop_episode |
| license | license string in the allowed list from configs/sources/<name>.yaml | drop_episode |
verify/hashing.py builds the benchmark hash index once (data/verified/benchmark_hashes.parquet) from the eval-only adapters and RoboRewardBench, and caches hashes per episode so reruns are cheap.
verify/manifest.py writes rows with the fields in the data model section plus norm_version and report_version (git hash of the repo at verification time). Splits are assigned by session using the adapter's splits(); the held-out id file is written once by tf manifest freeze-holdout and every later tf manifest run refuses to change a frozen row's split. Window sampling caps per source (max_windows, max_episodes) are config, applied by deterministic seeded sampling so a manifest is reproducible from the config and the raw data.

## Compute and environments

| Machine | Role | Notes |
| 5x AMD MI210 (64 GB each, gfx90a) | primary training, pseudo-labeling, eval | 27B sharded with FSDP, bf16 weights (about 11 GB per card) plus 15 to 20 GB activations with checkpointing; no 4-bit quantization (bitsandbytes on ROCm is unreliable). Target 8 to 10 s per 16k-token window across the cluster, about 5 h per epoch of 1,900 samples |
| Cloud RTX Pro 6000 (96 GB) | fallback for any GPU stage if ROCm blocks milestone 7 | bf16 LoRA fits with checkpointing; 18 to 22 s per window, 10 to 11 h per epoch |
| 8x RTX 2080 (8 to 11 GB, no bf16) | video decode, data conversion, verification, label scripts, CI | never load the 27B here |
| Baxter workstation | ROS 1, firmware flashing, recording, labeling | needs [baxter] extra only |
Known ROCm risk: Qwen3.5 uses Gated DeltaNet layers whose kernels come from the Triton-based flash-linear-attention library; Triton supports gfx90a but it is not the tested path. Milestone 7 (released model producing a parsable dense output on one window) is the gate; the README records the exact torch, ROCm, transformers and FLA versions that worked. Training cost model for planning: about 6NT FLOPs per sample with checkpointing, 2.6e15 for N = 27e9 and T = 16k; inference about a third of that.

## Prompts and training

prompts/flora.py reproduces the released FLORA-Rewarder prompts verbatim (system prompt, user prompt, answer grammar) and adds the force block in one place. Prompts are built from a Window and a force_mode; the function is the single source of truth for both training and eval, so the two can never drift.
Dense prompt: the paper's system prompt (64 frames at 4 FPS, 32 groups, retrospective, score scale 1/2/3, exact Worst / Quality / Rationales format) with one added sentence when force is present: A normalized fingertip or wrist force block follows the clip; each line gives per-arm grip force F, pad asymmetry D, its rate dF, gripper closure g, and contact flag c at the frame time. User prompt: Task: <task>, the multiview mosaic video, the force block, then the paper's clip description line with absolute times.
Outcome prompt: the paper's terminal-success system prompt in rationale mode for training (Rationale: ... Score: <1-5>) and score-only mode for benchmarks, with the same added force sentence. Video input follows the paper: uniformly sampled frames with first and last included, mosaic layout 1x1 to 2x2 by camera roles, source frame indices and fps passed to the Qwen video processor so timestamp tokens are absolute episode seconds.
prompts/parse.py parses model output into Label objects and rejects answers whose Worst disagrees with the minimum digit, replacing them with an all-normal window flagged parse_failed, exactly as the paper does.
train/dataset.py reads a manifest, builds prompts, tokenizes with the Qwen3.5 processor, masks everything but the answer tokens in the loss, and returns samples sorted by length for packing-free batching (batch size 1 per GPU, gradient accumulation to the configured effective batch).
train/sft.py is a plain Hugging Face training loop, not a framework: load the released checkpoint in bf16, keep Gated DeltaNet parameters in fp32 as the paper does, attach LoRA (rank 32, alpha 64) to the language model's attention and MLP projections, freeze the vision encoder and merger, FSDP across available GPUs with activation checkpointing, AdamW, cosine schedule with 3% warmup, one epoch, gradient clipping 1.0, output-token cross-entropy. It writes data/runs/<run_name>/ with the config copy, the manifest hash, the git hash, loss curves as CSV, and the adapter weights. It refuses to start unless holdout_ids.json exists and the manifest's report_version matches the current config.
Run matrix, each a YAML under configs/train/:
| Run | Config delta | Purpose |
| force | force_mode: real | the tactile model |
| control | force_mode: omitted, same seed, same manifest | matched control |
The two configs differ in exactly one key; a unit test diffs them to enforce it. Throughput target on 5x MI210: 8 to 10 s per window; tf train --dry-run tokenizes the full manifest, prints the token histogram and the estimated wall-clock, and exits.

## Evaluation harness

tf eval --checkpoint <path|released> --benchmark <name> --force-mode <mode> is one command that writes data/eval/<checkpoint>/<benchmark>/<mode>/predictions.parquet (one row per window or episode: ids, raw model text, parsed label, timing) and metrics.json. tf eval grid --config configs/eval/grid.yaml expands the full condition matrix and skips cells whose predictions already exist, so partial reruns are cheap.
Benchmarks are declared in configs/eval/benchmarks.yaml, each as a manifest name plus which metrics apply:
| Benchmark | Manifest | Prompts | Metrics |
| phantom_val | PHANTOM val split | dense + outcome | MAE, macro F1, balanced acc, bad-action F1, failure recall |
| reassemble_test | REASSEMBLE held-out sessions | dense + outcome | MAE, failure recall, bad-action F1 |
| hrdexdb | all trials with tactile (zero-shot) | outcome, score-only | accuracy and AUROC from E[s] vs binary label |
| softvtbench | all released demos (zero-shot) | dense + outcome | MAE vs DSR; failure recall on over-compression spans |
| baxter | Baxter human-labeled set | dense + outcome | all dense metrics, MAE, per-object breakdown |
| roboreward_500 | stratified 500-episode subsample, no force | outcome, score-only, omitted only | MAE vs the paper's 40.0 |
Conditions are the force modes from eval/ablations.py applied to each checkpoint: released (force omitted), control, and the force model under omitted, zero, real, shuffled, video_blank. The grid config lists which modes apply to which benchmark; roboreward_500 accepts only omitted.
eval/metrics.py implements the paper's definitions: dense metrics on 1 Hz stitched labels pooled over steps with classes 1/2/3; bad-action F1 is F1 of the non-normal class; failure recall is recall of class 1; outcome MAE on the 1 to 5 scale times 100; E[s] from the renormalized probabilities of the five score tokens (top-20 alternatives, keep digits 1 to 5). Every metric function takes two aligned dataframes and is unit-tested against hand-computed values.
Inference details that must match the paper: greedy decoding, thinking disabled, regex-constrained decoding for the dense grammar with at most 640 new tokens, 12 tokens in score-only mode, parse failures replaced by all-normal and counted in metrics.json under parse_failed.
Reporting: tf eval report renders one markdown table per benchmark with conditions as rows and metrics as columns, a per-object table for baxter, and a shuffle-delta column (real minus shuffled) so the channel-is-read check is visible without a second table.

## Baxter capture

Four thin-film force sensors, two per parallel gripper, read by one Teensy 4.0 and published into ROS on the robot clock, recorded alongside Baxter's cameras and gripper state into LeRobot v3 episodes that the baxter adapter reads like any other source. The sensor ordered from Amazon (ASIN B0CZ6L5NMM) must be confirmed against the spec in the open-questions section before the firmware constants are set.
Firmware, baxter/firmware/teensy_fsr/teensy_fsr.ino:
- Each FSR in a voltage divider with a fixed resistor R_ref (default 4.7 kΩ, set in the sketch header) to 3.3 V; wiper to analog pins A0 to A3 in the order L_a, L_b, R_a, R_b.
- 12-bit ADC, 16-sample oversampling, 200 Hz loop. Each tick emits one line over USB serial at 115200: <micros>,<a0>,<a1>,<a2>,<a3>\n. Raw counts only; no calibration in firmware.
- Responds to Z\n by capturing a 1 s baseline and echoing it; the ROS node calls this at the start of every episode with the gripper open and nothing in it.
ROS node, baxter/ros/fsr_node.py (ROS 1 on the Baxter workstation; a ROS 2 variant is the same class with a different transport):
- Reads serial, converts Teensy micros to ROS time with a linear fit refreshed every 5 s (offset and drift), and publishes /tactile/raw (Float32MultiArray, 4 counts) and /tactile/force (4 values in newtons from the calibration file) at 200 Hz, each stamped with the fitted ROS time.
- Publishes /tactile/health once per second: per-channel variance, saturation fraction, baseline drift. The recorder refuses to start an episode if any channel is flat or saturated.
baxter/calibrate.py: for each pad, press a known-weight stack (100 g to 2 kg) through the mounted puck and skin, record raw counts, fit a monotone piecewise-linear map counts to newtons, write configs/baxter_calibration.json with the fit, the date, and the R_ref used. Calibration is per pad and must be redone if a sensor is replaced or re-mounted.
Recorder, baxter/ros/recorder.py: subscribes to the scene camera and both wrist cameras (30 fps), /robot/end_effector/*/state (gripper position and command), /tactile/force, and the scripted-policy status topic; writes one episode directory per run with MP4 per camera, parquet for state and force, and a meta.json with object name, firmness level, policy parameters (grasp height, offset, commanded gripper force), and operator notes. The scripted policy publishes its parameters at episode start so they end up in meta.json without hand entry.
baxter/to_lerobot.py: converts the episode directories into a LeRobot v3 dataset with observation.images.{scene,wrist_left,wrist_right}, observation.state (Baxter joint and gripper state), observation.force (4 floats at 200 Hz, resampled to the 30 fps frame clock by the same force/resample.py), and the meta.json fields as episode-level metadata. Human labels are added later as labels.json (1 Hz quality digits plus outcome and object) by tf baxter label, a small CLI that plays the episode and takes keystrokes; it never shows the force plot.
Sync verification is the same sync check as every other source: the lag between force onset and the gripper command crossing closure must be under 100 ms per episode.

## Baxter evaluation protocol

Task: single-arm pick, carry, place and release over a four-step firmness ladder, scripted policy, 60 to 80 episodes, hand-labeled. The policy exists to produce a mix of successes and contact failures, not to be good.
| Object | Firmness | Expected contact failures |
| Wooden block | rigid | almost none (control) |
| Plastic cup | semi-rigid | slip on low grasp, mild deformation |
| Paper cup | compliant | crush near rim, slip |
| Empty can | thin-walled | crush, slip during carry |
Scripted policy parameters, randomized per episode and written to meta.json by the policy's status topic: grasp height, lateral offset, commanded gripper force, object identity and firmness level, optional wrist-camera occlusion flag. Targets for the batch: at least a third of windows contain a non-normal span; no single failure type above a third of failures; include a subset with the wrist camera partially occluded so the force-helps-when-vision-cannot-see case is testable.
Labeling (tf baxter label): two labelers, 1 Hz, classes 1/2/3 plus an outcome 1 to 5 per episode, judged from video and the physical result only; the tool never shows force. Disagreements are resolved together and the agreement rate is stored in labels.json. Labels derived from any force threshold are forbidden on this set because the programmatic detectors use the same thresholds in training.
Sensor hardware (plan decision): thin-film force-sensing resistors, 111 N class, about 10 mm sensing area, two per gripper; Tekscan FlexiForce A201-25 first choice, Interlink FSR X 402 or a generic RP-C18.3-ST film as fallbacks; six bought for four pads, one model on all four. Mount with double-sided tape, a 9 to 10 mm rigid puck over the sensing area and a 1 to 2 mm silicone skin; calibrate after mounting. Readout is the Teensy divider circuit in the capture section, never a one-sensor-per-hub kit.
Reporting on this set: every ablation condition, pooled and per object; the expected signature is no gain on the block and a widening gain down the ladder, plus a visible drop under force shuffle.

## Configuration, logging, testing, reproducibility

Configuration: every YAML under configs/ maps to a pydantic model in the owning stage (SourceConfig, ForceConfig, TrainConfig, EvalConfig, BaxterConfig). Models forbid extra keys so typos fail at load time. A config is loaded once by the CLI and passed down; stages never read YAML themselves. Every run directory gets a frozen copy of the resolved config.
CLI: tf is a typer app with one subcommand per stage. Every command prints the config path, the git hash, and the output path before doing work, and exits non-zero on any failed invariant. No command has side effects outside data/.
Logging: standard logging to stderr plus a JSONL log in the output directory. Training logs loss, learning rate, tokens per second and GPU memory every 10 steps; eval logs per-episode parse status. No experiment trackers in the core; a wandb hook is an optional [dev] extra that reads the JSONL.
Testing:
- tests/unit/ covers every public function with synthetic fixtures; the whole suite runs in under a minute on CPU with no datasets or GPU present.
- Round-trip tests: label spans to FLORA target string and back; prompt build to parse; window geometry against the paper's worked numbers (anchor schedule, group boundaries, dropped tail).
- Config tests: force.yaml and control.yaml differ in exactly one key; every configs/sources/*.yaml loads; every adapter is registered.
- An integration test tests/test_pipeline_smoke.py runs verify, manifest, dataset build and one eval cell on the fixtures with a stub model, so a broken stage fails CI before anyone touches the MI210s.
- Firmware has a host-side parser test on recorded serial lines.
Reproducibility: seeds for sampling, shuffling and ablation permutations are config values; the manifest is reproducible from config plus raw data; every artifact under data/ carries the git hash, config hash and manifest hash that produced it; tf eval report refuses to compare predictions whose manifest hashes differ. Environment is pinned by uv.lock; the ROCm and CUDA torch wheels are selected by an environment variable documented in the README, and the README records the exact versions that worked on each machine.

## Success criteria and expected results

What the implementer should expect, so a surprising number is investigated rather than reported:
- roboreward_500: released, control and force-omitted rows within about 3 MAE points of each other and of 40.0. A gain here points to contamination; a drop over 5 points means the LoRA pass hurt calibration (lower LR or shorter epoch, rerun).
- phantom_val: the force model should beat the control clearly on outcome MAE and failure recall; this is the headline number.
- reassemble_test: modest gain or parity.
- hrdexdb, softvtbench: zero-shot; any gain over control is a transfer result, parity is acceptable.
- baxter: a coin flip pooled; the per-object and occluded-subset breakdowns are where a real effect would show.
- Shuffle: failure recall of the force model must drop under shuffled; if it does not, the channel was ignored and that is the finding.
- Force-omitted row of the force model must match the control within noise; a gap means the block leaked into behavior when absent.
Gates, in order: milestone 7 parsable output on MI210s (else cloud); every training source passes verification and the holdout file is frozen before SFT starts; force-video sync under 100 ms on a Baxter episode before any Baxter recording counts; the released checkpoint's no-force rows on phantom_val and roboreward_500 exist before any trained-model row is read.
Follow-ons the code must not block: a learned force encoder consuming the per-group raw slices the resampler already returns; the 9B segmenter trained on SharpaDex subtask_index intervals (plus REASSEMBLE and AIRoA primitive boundaries) with the same force block; bimanual evaluation with a second arm's lines filled (RoboMIND for labels, AgiBot World 2026 for force); SoftVTBench and Isaac Lab simulation for crush labels at volume; the VLA shaping loop that turns dense q_t into per-window advantages.

## Implementation order

Build in this order; each milestone has an acceptance criterion that can be checked without the next one. Milestones 1 to 4 need no GPU and no datasets beyond fixtures.
1. Skeleton and schema. Package layout, schema.py, CLI stub, config models, CI running ruff, pyright, pytest. Accept: tf --help lists every subcommand; unit suite passes on fixtures.
2. Force channel. resample.py, normalize.py, render.py, ablation transforms. Accept: rendered block for a fixture matches the exact format in this spec byte for byte; token count asserted; normalization constants round-trip through norm.json.
3. Prompts and parsers. prompts/flora.py, prompts/parse.py, labels/windows.py. Accept: window geometry test reproduces the paper's anchor schedule; target-to-parse round trip on 1,000 random spans; force and control prompts differ only in the block.
4. Labels. mapping.py, events.py, precedence merge. Accept: each detector fires on its synthetic fixture and nowhere else; per-source mapping tables unit-tested.
5. Adapters, one at a time: phantom, sharpadex, rh20t, reassemble, airoa, then eval-only hrdexdb, softvtbench, baxter. Accept per adapter: discover() count matches the dataset card; one real episode loads on the episode clock; fixture test passes.
6. Verification and manifest. Checks, hashing, report, manifest writer, holdout freeze. Accept: tf verify produces a report for every training source; tf manifest writes rows; freeze-holdout then a second tf manifest leaves frozen splits untouched; overlap check removes a planted duplicate.
7. Released-model inference. Load FLORA-Rewarder, run the dense and outcome prompts on one PHANTOM window on the MI210s. Accept: parsable output; this is the ROCm gate.
8. Pseudo-labeling. Accept: SharpaDex windows get origin="pseudo" labels; refuses eval-only sources.
9. Training. dataset.py, sft.py, dry run, then the two runs. Accept: dry run reports token histogram and ETA; a 20-step smoke run on fixtures finishes on one GPU; loss decreases on the real manifest.
10. Eval harness. harness.py, metrics.py, grid, report. Accept: released checkpoint on phantom_val with omitted produces metrics; roboreward_500 MAE within 3 points of 40.0; grid resumes after interruption.
11. Baxter stack. Firmware, node, calibration, recorder, converter, labeler. Accept: /tactile/health is clean; calibration file written; one recorded episode converts and passes tf verify --source baxter including the sync check.
After milestone 11 the only remaining work is data: recording, labeling and running the grid. Milestones 1 to 6 and 11 can proceed in parallel with 7 to 10 on separate machines because they share only files.

## Assumptions and open questions

Assumptions the implementer may rely on until told otherwise:
- The released FLORA-Rewarder weights, processor and prompt templates are available from the Hugging Face nvidia/flora collection and load with the current transformers Qwen3.5 support.
- The five MI210s run ROCm with PyTorch, FSDP and the flash-linear-attention Triton kernels; if milestone 7 fails there, every GPU stage runs unchanged on a single 96 GB NVIDIA card with device_map sharding replaced by one device.
- Baxter runs ROS 1 with the standard baxter_interface topics; cameras are the head camera plus both wrist cameras at 30 fps.
- The scripted pick-and-place policy exists outside this repo and publishes its parameters on a topic; this repo only records it.
Open questions, each blocking one specific module:
- Sensor identity. Confirm the Amazon part (ASIN B0CZ6L5NMM) is a two-wire thin-film force-sensing resistor with roughly 10 mm sensing area and a range of at least 50 N; record its no-load resistance and the resistance at 10 N so R_ref in the firmware can be chosen to put a can-crush force near mid-scale. Blocks firmware constants and configs/baxter.yaml.
- PHANTOM license and the exact manifest fields (success, tactile_contact, peak force units). Blocks sources/phantom.py and the license check.
- RH20T port fidelity: whether observation.robot_ft in the community LeRobot ports is calibrated and which configs carry the fingertip tactile stream. Blocks sources/rh20t.py normalization.
- REASSEMBLE anomaly annotation format (span times vs frame indices; label vocabulary). Blocks labels/mapping.py.
- SoftVTBench FEM contact fields: which array gives per-finger contact force, and the deformation tolerance per object used for DSR. Blocks the softvtbench adapter.
- HRDexDB tactile layout per hand type and which hands to include. Blocks the hrdexdb adapter.
- Baxter gripper force command range and whether the scripted policy can vary it per episode. Blocks the recorder's meta.json fields.
Resolve each by reading the dataset card or a single episode and writing the answer into the adapter's module docstring, so the next reader doesn't repeat the investigation.