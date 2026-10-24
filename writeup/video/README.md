# Tactile FLORA project video

`tactile-flora.mp4` is a silent, three-minute walk through the project: the problem, the
idea, the force block, the controls, the four-stage process, the HRDexDB results and diagnostics,
the Baxter setup and result, caveats and conclusion. It follows the paper section by section.

Each of the 16 cards shows one slide of `writeup/slides/tactile-flora.pdf`, or one of the two
side-by-side clips (the standing-frame win from HRDexDB and Baxter episode 0), above a caption
band that carries the narration as on-screen text. Card length follows the caption's word count,
about 160 words per minute. There is no audio track. Every number in the captions is copied
from `writeup/tactile-flora-report.md`.

| File | What |
|---|---|
| `tactile-flora.mp4` | the video, 1920x1080, H.264, 24 fps, about 6 MB |
| `build_video.py` | the captions and the build: renders the slides with PyMuPDF, composes frames with Pillow, encodes and concatenates with the ffmpeg bundled in imageio-ffmpeg |

## Regenerate

```sh
uv venv /tmp/video-venv --python 3.11
uv pip install --python /tmp/video-venv/bin/python pymupdf pillow imageio-ffmpeg
/tmp/video-venv/bin/python writeup/video/build_video.py
```

Rebuild the slide PDF first if the deck changed (`writeup/slides/README.md`). The captions are
the `CARDS` list at the top of the script; the font is Droid Sans from the system font directory.
