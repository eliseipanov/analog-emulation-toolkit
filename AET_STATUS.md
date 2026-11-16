# AET — Actual Status (generated 2026-11-16)

## 1. Nodes

### AET Image Loader (`AET_ImageLoader`)
- INPUT_TYPES:
  - `image` (STRING, default "", image_upload=True) — drag & drop image from ComfyUI input folder
- RETURN_TYPES: ("IMAGE", "STRING") → image, file_path
- Description: Loads an image from the ComfyUI input directory, applies exif_transpose, converts to RGB, normalizes to float32 tensor in [0,1] range.
- Preset fields read/written: None

### AET Reference Loader (`AET_ReferenceLoader`)
- INPUT_TYPES:
  - `image` (STRING, default "", image_upload=True) — drag & drop reference photo
- RETURN_TYPES: ("IMAGE", "STRING", "STRING") → ref_image, ref_file_path, ref_EXIF_json
- Description: Loads a reference image, extracts its EXIF via piexif.load() and serializes to JSON via _exif_dict_to_json().
- Preset fields read/written: Reads EXIF from the reference image file; writes nothing to presets.

### AET Preset Loader (`AET_PresetLoader`)
- INPUT_TYPES:
  - `camera_model` (dropdown from presets/cameras/ + presets/custom/) — selects a camera preset
- RETURN_TYPES: ("STRING", "STRING") → exif_json, xmp
- Description: Loads a camera preset JSON file; if it contains an "exif" key uses that directly, otherwise converts via _camera_to_exif(). Returns EXIF JSON and any stored XMP string.
- Preset fields read/written: Reads preset JSON from presets/cameras/ and presets/custom/. Writes nothing.

### AET Preset Saver (`AET_PresetSaver`)
- INPUT_TYPES:
  - `ref_file_path` (STRING, forceInput=True) — path to reference image (from AET_ReferenceLoader)
  - `preset_name` (STRING, default "") — name of preset file (without .json)
  - `save_exif` (BOOLEAN, default False) — save EXIF tags from reference into preset
  - `save_profile` (BOOLEAN, default False) — save pixel profile (not implemented yet)
  - `overwrite` (BOOLEAN, default False) — if False, refuse to overwrite existing preset
  - `artist` (STRING, default "") — optional EXIF Artist field value
  - `copyright` (STRING, default "") — optional EXIF Copyright field value
- RETURN_TYPES: ("STRING",) → debug
- Description: Reads EXIF and XMP from the reference image file, optionally stores them plus artist/copyright into a preset JSON written to presets/custom/.
- Preset fields read/written: Reads EXIF + XMP from ref_file_path. Writes preset JSON (display_name, exif, xmp, Artist, Copyright) to presets/custom/.

### AET Pixel Emulator (`AET_PixelEmulator`)
- INPUT_TYPES:
  - `image` (IMAGE) — input image tensor
  - `scanner_model` (dropdown from presets/scanners/) — scanner preset selection
  - `frequency_cutoff` (FLOAT, default 0.2, min 0.05, max 0.4, step 0.01) — FFT cutoff radius
  - `grain_amount` (FLOAT, default 0.5, min 0.0, max 3.0, step 0.05) — synthetic grain strength
  - `row_shift` (FLOAT, default 0.1, min 0.0, max 1.0, step 0.01) — sub-pixel row shift
  - `lab_chroma_strength` (FLOAT, default 10.0, min 0.0, max 20.0, step 0.5) — LAB chroma noise strength
  - `enable_pixel_processing` (BOOLEAN, default True) — if False, passes image through unchanged
  - `seed` (INT, default 0, min 0, max 0xFFFFFFFF) — random seed
  - `hot_pixels` (INT, default 0, min 0, max 100) — number of random hot pixels
  - `dead_pixels` (INT, default 0, min 0, max 100) — number of random dead pixels
  - `dust_spots` (INT, default 0, min 0, max 50) — number of soft dust spots
  - `sensor_shading` (FLOAT, default 0.0, min 0.0, max 0.5, step 0.01) — sensor shading gradient
  - `chromatic_aberration` (FLOAT, default 0.0, min 0.0, max 2.0, step 0.05) — CA strength in pixels at edge
  - `vignette_strength` (FLOAT, default 0.0, min 0.0, max 0.5, step 0.01) — vignetting strength
  - `vignette_offset_x` (FLOAT, default 0.0, min -1.0, max 1.0, step 0.05) — vignette center X offset
  - `vignette_offset_y` (FLOAT, default 0.0, min -1.0, max 1.0, step 0.05) — vignette center Y offset
  - `is_grayscale` (BOOLEAN, default False) — disables LAB chroma noise and chromatic aberration
  - Optional: `file_path` (STRING, forceInput=True) — optional input image path
  - Optional: `ref_file_path` (STRING, forceInput=True) — optional reference image path for EXIF
- RETURN_TYPES: ("IMAGE", "STRING", "STRING") → image, debug_exif, exif_json
- Description: Applies pixel-level analog emulation (FFT grain, row shift, LAB chroma, sensor defects, optical defects) and generates EXIF from scanner preset or reference image.
- Preset fields read/written: Reads scanner preset EXIF via _scanner_to_exif() or reference EXIF. Writes nothing.

### AET Image Saver (`AET_ImageSaver`)
- INPUT_TYPES:
  - `image` (IMAGE) — input image tensor
  - `exif_json` (STRING, forceInput=True) — EXIF JSON from AET_PixelEmulator or AET_PresetLoader
  - `filename_prefix` (STRING, default "SCAN") — output filename prefix
  - `output_subfolder` (STRING, default "scans") — subfolder inside ComfyUI/output/
  - `quality` (INT, default 88, min 1, max 100) — JPEG quality
  - `subsampling` (["4:4:4", "4:2:2", "4:2:0"], default "4:4:4") — JPEG chroma subsampling
  - `date_override` (STRING, default "") — optional date override (YYYY:MM:DD HH:MM:SS)
  - `overwrite_artist_copyright` (BOOLEAN, default False) — if True, overwrite Artist/Copyright in EXIF
  - `artist` (STRING, default "") — optional Artist value
  - `copyright` (STRING, default "") — optional Copyright value
  - `overwrite_serial` (BOOLEAN, default False) — if False, removes serial numbers; if True, generates random ones
  - `xmp` (STRING, default "") — optional raw XMP metadata to write
- RETURN_TYPES: ("STRING",) → saved_path
- Description: Saves the image as JPEG with EXIF metadata. Refuses to save without EXIF. Updates date, serial numbers, dimensions, Artist/Copyright, and synchronizes XMP date/serial numbers with EXIF.
- Preset fields read/written: Reads EXIF JSON and XMP string. Writes JPEG file to ComfyUI/output/<subfolder>/.

## 2. Preset format

### presets/scanners/brother_ads1700w.json (first scanner preset)
```json
{
  "display_name": "[Flatbed] Brother ADS-1700W",
  "make": "Brother",
  "model": "ADS-1700W",
  "software": "ControlCenter",
  "file_source": null,
  "exif_mode": "minimal",
  "filename_prefix": "Scan_",
  "filename_pattern": "{prefix}{counter}"
}
```

Top-level keys:
- `display_name` — human-readable preset name shown in dropdown
- `make` — scanner manufacturer string → EXIF ImageIFD.Make
- `model` — scanner model string → EXIF ImageIFD.Model
- `software` — software name → EXIF ImageIFD.Software
- `file_source` — FileSource EXIF tag value (null in this preset)
- `exif_mode` — "minimal" reduces EXIF to only Software + DateTime
- `filename_prefix` — suggested filename prefix for output
- `filename_pattern` — suggested filename pattern

### presets/cameras/canon_eos_r6.json (first/only camera preset)
```json
{
  "display_name": "Canon EOS R6",
  "make": "Canon",
  "model": "Canon EOS R6",
  "lens_model": "RF24-70mm F2.8 L IS USM",
  "software": "Adobe Lightroom 13.2",
  "iso": 400,
  "exposure_time": "1/250",
  "f_number": 2.8,
  "focal_length": "50",
  "white_balance": "Auto",
  "color_space": 1
}
```

Top-level keys:
- `display_name` — human-readable preset name shown in dropdown
- `make` — camera manufacturer → EXIF ImageIFD.Make
- `model` — camera model → EXIF ImageIFD.Model
- `lens_model` → EXIF ExifIFD.LensModel
- `software` → EXIF ImageIFD.Software
- `iso` → EXIF ExifIFD.ISOSpeedRatings
- `exposure_time` → EXIF ExifIFD.ExposureTime (parsed as rational)
- `f_number` → EXIF ExifIFD.FNumber
- `focal_length` → EXIF ExifIFD.FocalLength
- `white_balance` → EXIF ExifIFD.WhiteBalance (0=Auto, 1=manual)
- `color_space` → EXIF ExifIFD.ColorSpace

### presets/custom/*.json
No preset JSON files exist in presets/custom/ (only `.gitkeep`). No examples available.

## 3. processing.py

Functions:

- `_fft_add_grain(channel, cutoff, grain_amount, rng)` — Adds synthetic grain to high frequencies via FFT; preserves original low frequencies. **DOES FFT/spectral analysis.**
- `_grain_luminance_mask(gray)` — Returns a luminance-based mask (clip of 1 - |gray - 0.45| * 2) for spatial noise scaling.
- `_subpixel_row_shift(img, max_shift, rng)` — Applies horizontal sub-pixel shift per row to simulate scanner head jitter.
- `_lab_chroma_shift(img, strength, rng)` — Adds noise to LAB a/b channels and converts back to RGB.
- `_apply_hot_pixels(img, count, rng)` — Sets `count` random pixels to white (1.0).
- `_apply_dead_pixels(img, count, rng)` — Sets `count` random pixels to black (0.0).
- `_apply_dust_spots(img, count, rng)` — Draws `count` soft-edged semi-transparent Gaussian-blurred circles that darken the image.
- `_apply_sensor_shading(img, strength, rng)` — Applies a smooth low-frequency brightness gradient (multiplicative factor between 1.0 and 1.0-strength).
- `_apply_chromatic_aberration(img, strength, rng)` — Shifts R channel outward and B channel inward radially from center using bilinear interpolation (map_coordinates).
- `_apply_offcenter_vignette(img, strength, offset_x, offset_y)` — Applies off-center radial darkening from 1.0 at center to (1.0-strength) at farthest corner.
- `apply_film_scan_processing(image, frequency_cutoff, grain_amount, seed, row_shift=0.0, lab_chroma_strength=0.0, hot_pixels=0, dead_pixels=0, dust_spots=0, sensor_shading=0.0, chromatic_aberration=0.0, vignette_strength=0.0, vignette_offset_x=0.0, vignette_offset_y=0.0, is_grayscale=False)` — Main pipeline: FFT grain → row shift → LAB chroma → sensor defects → optical defects. Returns stacked tensor.

Functions that do FFT/spectral/noise analysis: `_fft_add_grain` (uses np.fft.fft2/ifft2 for frequency-domain grain synthesis).

## 4. Features added beyond README

The following features exist in code but are NOT documented in README.md:

- Sensor defect simulation: `hot_pixels`, `dead_pixels`, `dust_spots`, `sensor_shading` inputs on AET_PixelEmulator
- Optical defect simulation: `chromatic_aberration`, `vignette_strength`, `vignette_offset_x`, `vignette_offset_y` inputs on AET_PixelEmulator
- Grayscale mode: `is_grayscale` input on AET_PixelEmulator (disables LAB chroma noise and chromatic aberration)
- XMP metadata: reading via `img.getxmp()` in AET_PresetSaver, XMP output from AET_PresetLoader, XMP input/write in AET_ImageSaver with `xmp=` parameter
- XMP synchronization: AET_ImageSaver updates XMP CreateDate/DateCreated to match EXIF date, and SerialNumber/LensSerialNumber when overwrite_serial is True
- Serial number control: `overwrite_serial` input on AET_ImageSaver (random serial generation preserving digit/letter classes); serial numbers removed when False
- Artist/Copyright control: `overwrite_artist_copyright`, `artist`, `copyright` inputs on AET_ImageSaver; `artist`/`copyright` inputs on AET_PresetSaver
- EXIF dimension override: AET_ImageSaver sets PixelXDimension/PixelYDimension from actual image tensor shape
- MakerNote stripping: explicitly excluded from JSON round-trip in _exif_dict_to_json/_json_to_exif_dict
- GPSVersionID special handling: stored as list in JSON, restored as tuple
- Frontend upload button: web/js/upload_button.js adds upload button and preview for AET_ImageLoader and AET_ReferenceLoader nodes

## 5. Known gaps / TODOs

No TODO/FIXME/XXX comments found in the codebase.

Potential gaps observed (not marked in code):
- `save_profile` input on AET_PresetSaver exists but profile saving is not implemented (debug line says "Profile saving is not implemented yet. Skipped.")
- `_camera_to_exif` uses `eval()` on `preset["exposure_time"]` which could be a security risk with untrusted preset files
- No camera presets beyond Canon EOS R6; no film scanner presets beyond Brother ADS-1700W
- No EXIF GPS data handling beyond stripping GPSTag from JSON round-trip