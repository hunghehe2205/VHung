# inference_viz

Qualitative inference + visualization for the UCF-Crime VadCLIP model.

## Run

Paths (`MODEL_PATH`, `FEATURE_ROOT`, `RAW_VIDEO_ROOT`, `TEST_LIST`, `GT_PATH`, `OUTPUT_DIR`) are **hardcoded as module-level constants at the top of `infer.py`** — edit them to match the server before running.

Everything else stays on the CLI:

```bash
# single video
python -m inference_viz.infer --video Arrest001_x264

# full test set
python -m inference_viz.infer

# with overrides
python -m inference_viz.infer --select-frames 20 --tau 0.2
```

Feature path is resolved as `<FEATURE_ROOT>/<label>/<basename-of-csv-path>`; raw video as `<RAW_VIDEO_ROOT>/<label>/<video_id>.mp4`. This makes the CSV's absolute path column only used for its filename portion.

## Output

```
ucf_infer_results/
└── <video_id>/
    ├── viz.png         # K sampled frames on top + anomaly curve (GT shaded, red vlines at samples)
    └── metadata.json   # video info, per-snippet scores, sampled snippet & frame indices, GT segments
```

## Key args

| Flag | Default | Meaning |
|---|---|---|
| `--model-path` | `model/model_ucf.pth` | CLIPVAD checkpoint |
| `--test-list` | `list/ucf_CLIP_rgbtest.csv` | Feature CSV |
| `--raw-video-root` | `/home/emogenai4e/emo/Hung_data/UCF_Crime` | Root containing `<Class>/<id>.mp4` |
| `--gt-path` | `data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | GT temporal segments |
| `--output-dir` | `ucf_infer_results` | Where per-video folders are written |
| `--video` | (none) | Run a single video by id (e.g. `Arrest001_x264`) |
| `--select-frames` | 16 | K snippets picked by density-aware sampling |
| `--tau` | 0.1 | Smoothing added to every snippet score before sampling |

Sampling operates at the snippet level (1 snippet = 16 raw frames). The frame chosen for each sampled snippet is its temporal center (`snippet_idx * 16 + 8`).
