# Quickstart

Short tour of the app once it's built. For install steps see [INSTALL.md](INSTALL.md); for architecture see [README.md](README.md).

## Launch

```bash
./build/bin/lightroom_clone
```

The window opens with an empty viewport on the left and a stack of collapsible effect panels on the right.

## Open an image

- **File → Open Image…** (`Ctrl+O`)
- Supported: `.png`, `.jpg/.jpeg`, `.bmp`, `.tif/.tiff`, and — when LibRaw is present — `.cr2 .cr3 .nef .nrw .arw .dng .raf .orf .rw2`
- RAW files are decoded into 16-bit RGBX64 so highlight/shadow recovery has real headroom

## Pan / zoom

| Action | Shortcut |
|---|---|
| Zoom in | `+` / `=` |
| Zoom out | `-` |
| Fit to window | `Ctrl+0` |
| 100 % (1 image px : 1 screen px) | `Ctrl+1` |
| Pan | middle-mouse drag |

Zoom range is 1× (fit) to 64×.

## Editing

Right panel, top-to-bottom pipeline order:

1. **Hot Pixel** — single-pixel outlier removal
2. **Exposure** — EV stops
3. **White Balance** — temperature / tint
4. **Brightness** — brightness + contrast
5. **Saturation** — saturation + vibrancy (skin-tone protected)
6. **Blur** — Gaussian / Box, radius
7. **Grayscale** — luminosity, gated by an internal checkbox
8. **Unsharp** — amount / radius / threshold
9. **Denoise** — bilateral
10. **Vignette** — amount / midpoint / roundness
11. **Film Grain** — fBm noise, image-anchored, seed control
12. **Split Toning** — shadows / highlights hue + saturation
13. **Clarity** — local-contrast midtone boost
14. **Color Balance** — per-zone RGB shifts

Each panel has a collapse toggle in its header.

### Sliders

Every numeric parameter uses a `ParamSlider` (label + slider + spinbox):

- drag the slider for a live preview
- release (or commit the spinbox) to trigger a full GPU reprocess
- **double-click** the slider to reset to 0

### Enabling / disabling effects

**View → Effects** — check/uncheck individual effects. Unchecking hides the panel *and* skips that effect in the pipeline.

## GPU device

A combo box at the top of the right panel lists every OpenCL device found at startup. Switching it reinitialises all effect kernels on the new device and triggers a full reprocess. Hover the combo for a tooltip.

## Save

- **File → Save Image…** (`Ctrl+S`) — saves the current fully-processed output at source resolution
- Format is inferred from the chosen extension

## Quit

**File → Quit** (`Ctrl+Q`). Window geometry is persisted via `QSettings`.
