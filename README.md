# Analog Emulation Toolkit (AET)

A ComfyUI post-processing toolkit for emulating analog photo sources — film scanners and digital cameras — by combining pixel-level grain synthesis with realistic EXIF metadata.

## What it does

- **Pixel emulation**: FFT-based grain, sub-pixel row shift, and LAB chroma noise to make generated images look like they came from a real sensor.
- **EXIF emulation**: generate scanner/camera EXIF from presets, or save a preset from a real reference photo.
- **Preset system**: JSON presets for scanners, cameras, and user-saved custom presets.
- **Mandatory EXIF**: the saver refuses to write a file without EXIF, so no "naked" JPEGs are produced.

## Nodes

| Node | Purpose |
|---|---|
| `AET Image Loader` | Load a working image from the ComfyUI input folder. |
| `AET Reference Loader` | Load a reference photo (real camera/scanner) and extract its EXIF. |
| `AET Preset Loader` | Select a scanner or camera preset and output its EXIF as JSON. |
| `AET Preset Saver` | Save EXIF (and later pixel profile) from a reference photo into `presets/custom/`. |
| `AET Pixel Emulator` | Apply pixel-level analog emulation and generate EXIF from a scanner preset. |
| `AET Image Saver` | Save the result as JPEG with EXIF. Refuses to save without EXIF. |

## Installation

### Via ComfyUI Manager

Search for **Analog Emulation Toolkit** in ComfyUI Manager and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:eliseipanov/analog-emulation-toolkit.git
cd analog-emulation-toolkit
pip install -r requirements.txt
```
Restart ComfyUI.

## Requirements
ComfyUI (recent version with new frontend)

Python 3.10+

numpy, scipy, scikit-image, piexif, Pillow

## Presets
Presets are stored as JSON in:
```
presets/
├── scanners/   # built-in scanner presets
├── cameras/    # built-in camera presets
└── custom/     # user-saved presets (created by AET Preset Saver)
```
## Typical workflows
Scanner emulation:

```
Load Image → AET Pixel Emulator (scanner preset) → AET Image Saver
```
## Camera EXIF only (no pixel processing):
```
Load Image → AET Preset Loader (camera preset) → AET Image Saver
```
## Save a preset from a real photo:

```
AET Reference Loader → AET Preset Saver (enable save_exif)
```
## Notes
AET Image Saver always overwrites the date in EXIF with the current date unless date_override is set.

Orientation is forced to 1 on save, because pixels are already oriented correctly by the loaders.

MakerNote is stripped from presets because it cannot survive JSON round-trip and becomes invalid after any modification.

## License

MIT — see [LICENSE](LICENSE).
