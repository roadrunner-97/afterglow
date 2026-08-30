# Quickstart

Short tour of the app once it is installed. For build and dependency steps see
[INSTALL.md](INSTALL.md).

## Launch

```bash
./build/bin/afterglow
```

Pass an image path to open it directly in Develop:

```bash
./build/bin/afterglow /path/to/photo.raw
```

Afterglow opens in Gallery view. Use the mode buttons in the toolbar to move
between Gallery, Loupe, and Develop.

## Open an image

- **File → Open Folder…** (`Ctrl+Shift+O`) opens a folder in Gallery
- **File → Open Image…** (`Ctrl+O`) opens one image directly in Develop
- Supported: `.png`, `.jpg/.jpeg`, `.bmp`, `.tif/.tiff`, `.cr2`, `.cr3`,
  `.nef`, `.nrw`, `.arw`, `.dng`, `.raf`, `.orf`, and `.rw2`
- RAW files are decoded into 16-bit RGBX64 so highlight/shadow recovery has real headroom

## Gallery and Loupe

Gallery builds cached thumbnails and edited previews in the background. Scroll
to resize thumbnails, use the arrow keys to select, and press `Enter` (or
double-click) to open Loupe. Mark a photo with `A` (Accept), `R` (Refine), or
`D` (Decline); pressing the active mark again clears it.

Loupe provides a large preview, mark controls, image-version choices, and
camera metadata. Use `Left`/`Right` to navigate, `F` to fit the image, and
`Enter` to move into Develop.

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

Develop has metadata and history on the left, the image viewport in the centre,
and GPU and effect controls on the right. The pipeline order is:

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

Each panel has a collapse toggle in its header. Crop & Rotate comes before the
effects listed above.

### Sliders

Every numeric parameter uses a `ParamSlider` (label + slider + spinbox):

- drag the slider for a live preview
- release (or commit the spinbox) to trigger a full GPU reprocess
- double-click a slider to restore that parameter's default value

### Enabling / disabling effects

**View → Effects** — check/uncheck individual effects. Unchecking hides the panel *and* skips that effect in the pipeline.

## GPU device

A combo box at the top of the right panel lists every OpenCL device found at
startup, including CPU runtimes such as POCL. Switching devices reinitialises
the kernels and triggers a full reprocess.

## Save

- **File → Save Image…** (`Ctrl+S`) opens the export dialog
- Choose JPEG, PNG, or TIFF, output quality, a naming suffix, resize mode, and
  what to do when the destination already exists
- Crop and rotation are baked into the exported full-resolution image

## Editing shortcuts

| Action | Shortcut |
|---|---|
| Undo / redo | `Ctrl+Z` / `Ctrl+Shift+Z` or `Ctrl+Y` |
| Copy / paste develop settings | `Ctrl+C` / `Ctrl+V` |
| Hold before-edits preview | `\` |

## Quit

**File → Exit** (`Ctrl+Q`). Window geometry and the last-used directory are
remembered between sessions.
