# Local Adjustments Implementation Plan

Local adjustments will be delivered as tested vertical slices, beginning with
one linear-gradient mask controlling exposure. Masks are defined in normalized
coordinates on the uncropped source image. The viewport transforms their
controls for display, so edits remain stable through pan, zoom, crop, rotation,
preview resizing, export, and reopening.

## Target model

Each local adjustment has a stable identity, name, enabled state, mask, and a
set of adjustment parameters. A mask owns its source-space geometry, feather,
and inversion state. Linear and radial masks are evaluated analytically on the
GPU; brush masks may later own tiled raster data.

## Milestones

### Related viewport follow-up

- Allow zooming below fit-to-window scale so a configurable margin of canvas
  is visible around the entire image. This makes edge-to-edge crop placement
  easier to inspect.
- Keep fit-to-window as the Ctrl+0 target; zooming out is an intentional user
  state, not a change to fit behavior.
- Crop and mask coordinates remain clamped to the source image even while the
  surrounding canvas is visible.
- Preserve cursor-centered wheel zoom and ensure rotated image corners can all
  be brought into view.

### 0. Linear-gradient interaction prototype

Status: accepted after human review. On-canvas tools have exclusive ownership;
the gradient uses a transparent-to-blue feather visualization whose affected
side visibly swaps when inverted.

- Drag to create a gradient.
- Display start, midpoint, and end handles.
- Move the gradient from its center handle.
- Change direction and feather width with the outer handles.
- Support selection, cancellation, deletion, inversion, reset, and overlay
  visibility without modifying pixels yet.
- Exercise crop, rotation, pan, zoom, and viewport-edge behavior.

Human acceptance gate:

- Test with both mouse and trackpad on bright, dark, detailed, and low-contrast
  photographs.
- A new user can create and reposition a gradient without instruction.
- The affected side and feather region are immediately apparent.
- Handles remain visible without obscuring the photograph.
- Dragging has no visible lag, jumps, or geometry drift.
- Creating or selecting a mask is not easily confused with panning.
- Escape cancels an unfinished gesture and Delete removes the selected mask.
- Cursor feedback explains which parts can move, rotate, or resize.
- Controls remain usable near image and viewport boundaries.

Decline the UI if users repeatedly grab the wrong handle, cannot predict the
affected side, create masks while trying to select or pan, see geometry shift
after zoom or rotation, or encounter sticky or delayed manipulation.

### 1. Core mask model

Status: complete for linear gradients. The ordered adjustment stack assigns
stable IDs and is covered independently of UI and GPU code.

- Add processing-independent mask and adjustment types.
- Represent a linear mask with a center, direction toward the affected side,
  feather half-width, and inversion flag.
- Evaluate a deterministic weight from 0 to 1 in normalized source space.
- Cover horizontal, vertical, diagonal, inverted, degenerate, narrow, and
  out-of-bounds geometry with unit tests.

### 2. Serialization and history

Status: complete for the current single-gradient slice. Version-2 sidecars
round-trip local adjustment state, legacy histories learn the new synthetic
state domain without losing their log, and committed mask/exposure gestures
participate in the normal chronological undo/redo stream.

- Version the sidecar representation and preserve stable adjustment IDs.
- Round-trip mask geometry, parameters, ordering, names, and enabled state.
- Continue loading existing sidecars and safely ignore unknown future fields.
- Make creation, deletion, movement, inversion, and parameter changes undoable.
- Coalesce a continuous pointer drag into one history entry.

Automated acceptance includes old-sidecar compatibility, lossless round trips,
safe malformed-input handling, undo/redo for every operation, and one history
entry per gesture rather than per mouse event.

### 3. GPU exposure slice

Status: complete for linear exposure. Commit, live preview, pan/zoom cache,
export, and background proofs share the analytical OpenCL mask evaluation.

- Transport mask descriptions through the shared OpenCL pipeline.
- Apply local exposure using the analytical mask weight.
- Support multiple, inverted, overlapping, and disabled masks.
- Avoid image upload/readback per mask and avoid kernel recompilation per edit.
- Match a CPU reference evaluator for RGB32 and RGBX64 inputs.
- Add golden tests for preview/export parity and benchmarks for 1, 5, and 20
  masks on large images.

### 4. Production linear-gradient UI

Status: implemented for human review. A mask layer can currently combine
Exposure (including tonal zones), Saturation & Vibrancy, and Grayscale. Effect
panels switch explicitly between preserved global parameters and the selected
mask's parameters; the same generic per-effect map is persisted and used by
preview, proof, and export.

- Add a Local Adjustments panel and adjustment list.
- Add, rename, enable, invert, duplicate, reorder, and delete gradients.
- Expose local exposure and overlay visibility controls.
- Show the selected mask clearly and other masks unobtrusively.
- Preview continuously while dragging and commit history on release.

Human release gate:

Use at least three reviewers when possible: an experienced Lightroom or
Capture One user, an Afterglow user uninvolved in implementation, and someone
new to local photo adjustments. Ask each to brighten a face, darken a sky,
invert and soften a gradient, temporarily disable it, manage two adjustments,
undo a deletion, crop and rotate, reopen the image, and compare an export.

Accept when at least two of three complete every task without help, nobody
silently edits the wrong mask, selection and affected-region feedback are
clear, interaction stays responsive, undo boundaries feel natural, and the
reopened and exported results match Develop. Validate at 100%, 150%, and 200%
display scaling.

Decline if users lose adjustments, edit the wrong mask unnoticed, cannot judge
the result without constantly hiding the overlay, experience lag that feels
broken, receive surprising undo behavior, or see preview/export or crop/rotate
disagreement. Record completion, errors, and hesitation rather than relying on
general preference questions.

### 5. Radial gradients

- Add an elliptical mask with center, axes, rotation, feather, and inversion.
- Reuse the established adjustment list, parameters, persistence, history, GPU
  transport, and selection behavior.

Human review concentrates on distinguishing move, resize, rotate, and feather
gestures; communicating inside versus outside; small-mask usability; and
consistent modifier keys. Decline if resize regularly becomes rotation or the
affected region is unclear without toggling inversion.

### 6. Additional parameters

Exposure (including highlights and shadows), Saturation & Vibrancy, and
Grayscale are the first review slice. Follow with temperature/tint, clarity,
sharpness, and denoise. Define how each composes with its global counterpart
and test neutral values, overlaps, both pixel formats, and preview/export
parity. Neighborhood operations require deliberate pipeline design so they do
not multiply blur passes or GPU transfers.

Humans must review temperature, clarity, sharpening, and denoise for natural
transitions, halos, banding, and agreement between fit view, 100% view, and
export.

### 7. Brush masks

Begin with a fixed-opacity brush and eraser. Add stroke interpolation, size,
hardness, per-stroke undo, incremental GPU updates, resolution-independent or
tiled persistence, and bounded history memory. Flow and tablet pressure follow
only after the base stroke model is reliable.

Accept when strokes stay under the pointer at speed, have no dots or gaps,
density is independent of event frequency, zoom does not alter saved geometry,
undo works per stroke, and latency is low on the minimum supported machine.
Feathered 16-bit exports must not band.

### 8. Advanced masks

After manual masks mature, add luminance and color ranges, mask intersection
and subtraction, subject/sky detection, and refinement. Automatic detection
must produce an editable mask rather than a baked result.

## Invariants for every milestone

- Sidecars stay non-destructive and backward compatible.
- Preview, proof, and export use identical mask semantics.
- Geometry survives pan, zoom, crop, rotation, resizing, and reopening.
- Device changes do not lose or corrupt edits.
- Live updates are coalesced without changing the committed result.
- Existing images without masks incur negligible overhead.
- No mask causes a separate image upload or readback.
- Every committed operation is reversible through undo.
- Invalid or unsupported masks fail closed.

Do not start radial or brush UI work until the production linear-gradient UI
passes its human gate. Interaction problems found later would otherwise be
replicated across every mask type.
