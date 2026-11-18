# Music Viewer

A native 3D music visualizer for Windows. It listens to **whatever your PC is playing**
(Spotify, YouTube, games — anything on the default output device) via WASAPI loopback
and renders it in real time with OpenGL.

- A glowing blob that swells with the bass and ripples with the mids
- A wireframe shell that spins with the treble
- A ring of 128 frequency bars (log-spaced, mirrored) with floor reflections
- A drifting particle field, beat-synced pulses and HDR bloom

## Run

Double-click `run.bat` (it creates the virtual environment on first run), or:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py              # --fullscreen to start fullscreen
```

Requires Python 3.10+ and an OpenGL 3.3 capable GPU.

## Controls

| Key | Action |
| --- | --- |
| Mouse drag / wheel | Orbit / zoom the camera |
| `F` / `F11` | Toggle fullscreen |
| `C` | Next color palette (Aurora, Sunset, Toxic, Ice, Rainbow) |
| `R` | Toggle auto-orbit |
| `Up` / `Down` | Sensitivity |
| `[` / `]` | Bloom strength |
| `H` | Show / hide help |
| `Esc` / `Q` | Quit |

If you switch your default playback device in Windows, the visualizer follows it within a couple of seconds.

## Code layout

- `audio.py` — loopback capture thread + FFT (Blackman window, -100..-30 dB → 0..1)
- `scene.py` — geometry, GLSL shaders, MSAA render, bloom (downsample/upsample chain), HUD overlay
- `main.py` — window, input, and how audio levels drive the visuals (beat detection, smoothing, palettes)
