import base64
import torch
import numpy as np
import piexif
import folder_paths
import os
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps

from .processing import apply_film_scan_processing


_SCANNERS_DIR = Path(__file__).parent / "presets" / "scanners"
_CAMERAS_DIR = Path(__file__).parent / "presets" / "cameras"
_CUSTOM_DIR = Path(__file__).parent / "presets" / "custom"


def _list_presets_from(directory):
    presets = []
    if directory.exists():
        for f in sorted(directory.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                presets.append(data.get("display_name", f.stem))
            except Exception:
                presets.append(f.stem)
    return presets if presets else ["None"]


def _load_preset_from(directory, display_name):
    if not directory.exists():
        return None
    for f in sorted(directory.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("display_name") == display_name:
                return data
        except Exception:
            continue
    return None


def _list_scanners():
    return _list_presets_from(_SCANNERS_DIR)


def _list_cameras():
    presets = []
    for directory in (_CAMERAS_DIR, _CUSTOM_DIR):
        if directory.exists():
            for f in sorted(directory.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    presets.append(data.get("display_name", f.stem))
                except Exception:
                    presets.append(f.stem)
    return presets if presets else ["None"]


def _load_scanner(display_name):
    return _load_preset_from(_SCANNERS_DIR, display_name)


def _load_camera(display_name):
    for directory in (_CAMERAS_DIR, _CUSTOM_DIR):
        preset = _load_preset_from(directory, display_name)
        if preset:
            return preset
    return None


def _scanner_to_exif(preset):
    now = datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode()
    zeroth = {
        piexif.ImageIFD.Make: preset.get("make", "").encode(),
        piexif.ImageIFD.Model: preset.get("model", "").encode(),
        piexif.ImageIFD.Software: preset.get("software", "").encode(),
        piexif.ImageIFD.DateTime: now,
    }
    exif = {
        piexif.ExifIFD.DateTimeOriginal: now,
        piexif.ExifIFD.DateTimeDigitized: now,
    }
    if preset.get("file_source") is not None:
        exif[piexif.ExifIFD.FileSource] = str(preset["file_source"]).encode()

    if preset.get("exif_mode") == "minimal":
        zeroth = {
            piexif.ImageIFD.Software: preset.get("software", "").encode(),
            piexif.ImageIFD.DateTime: now,
        }

    return {"0th": zeroth, "Exif": exif, "GPS": {}, "1st": {}}


def _camera_to_exif(preset):
    now = datetime.now().strftime("%Y:%m:%d %H:%M:%S").encode()
    zeroth = {
        piexif.ImageIFD.Make: preset.get("make", "").encode(),
        piexif.ImageIFD.Model: preset.get("model", "").encode(),
        piexif.ImageIFD.Software: preset.get("software", "").encode(),
        piexif.ImageIFD.DateTime: now,
    }
    exif = {
        piexif.ExifIFD.DateTimeOriginal: now,
        piexif.ExifIFD.DateTimeDigitized: now,
        piexif.ExifIFD.ColorSpace: preset.get("color_space", 1),
    }
    if preset.get("lens_model"):
        exif[piexif.ExifIFD.LensModel] = preset["lens_model"].encode()
    if preset.get("iso"):
        exif[piexif.ExifIFD.ISOSpeedRatings] = preset["iso"]
    if preset.get("exposure_time"):
        exif[piexif.ExifIFD.ExposureTime] = (1, int(1 / float(eval(preset["exposure_time"])) if "/" in preset["exposure_time"] else 1))
    if preset.get("f_number"):
        exif[piexif.ExifIFD.FNumber] = (int(preset["f_number"] * 10), 10)
    if preset.get("focal_length"):
        exif[piexif.ExifIFD.FocalLength] = (int(preset["focal_length"]), 1)
    if preset.get("white_balance"):
        exif[piexif.ExifIFD.WhiteBalance] = 0 if preset["white_balance"] == "Auto" else 1

    return {"0th": zeroth, "Exif": exif, "GPS": {}, "1st": {}}


def _exif_dict_to_json(exif_dict):
    serializable = {}
    for ifd in ("0th", "Exif", "GPS", "1st"):
        serializable[ifd] = {}
        for tag, value in exif_dict.get(ifd, {}).items():
            tag_info = piexif.TAGS[ifd].get(tag)
            if tag_info is None:
                continue
            tag_name = tag_info["name"]
            tag_type = tag_info["type"]

            if tag_name == "MakerNote":
                continue
            if tag_name in ("ExifTag", "GPSTag", "InteroperabilityTag"):
                continue

            if tag_type in (1, 7):  # BYTE, UNDEFINED
                if isinstance(value, bytes):
                    serializable[ifd][tag_name] = base64.b64encode(value).decode("ascii")
                elif isinstance(value, list) and all(isinstance(v, int) for v in value):
                    serializable[ifd][tag_name] = base64.b64encode(bytes(value)).decode("ascii")
                elif isinstance(value, int):
                    serializable[ifd][tag_name] = base64.b64encode(bytes([value])).decode("ascii")
                else:
                    serializable[ifd][tag_name] = str(value)
            elif tag_type == 2:  # ASCII
                if isinstance(value, bytes):
                    serializable[ifd][tag_name] = value.decode("utf-8", errors="replace")
                else:
                    serializable[ifd][tag_name] = str(value)
            else:
                # RATIONAL, SRATIONAL, SHORT, LONG, etc. JSON handles tuples/lists as arrays
                serializable[ifd][tag_name] = value
    return json.dumps(serializable, ensure_ascii=False)


def _json_to_exif_dict(exif_serializable):
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag_name, value in exif_serializable.get(ifd, {}).items():
            if tag_name in ("MakerNote", "ExifTag", "GPSTag", "InteroperabilityTag"):
                continue

            tag_id = None
            tag_type = None
            for tid, tinfo in piexif.TAGS[ifd].items():
                if tinfo["name"] == tag_name:
                    tag_id = tid
                    tag_type = tinfo["type"]
                    break

            if tag_id is None:
                continue

            if tag_type in (5, 10):  # RATIONAL, SRATIONAL
                if isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], list):
                        exif_dict[ifd][tag_id] = tuple(tuple(v) for v in value)
                    else:
                        exif_dict[ifd][tag_id] = tuple(value)
                else:
                    exif_dict[ifd][tag_id] = value
            elif tag_type in (1, 7):  # BYTE, UNDEFINED
                if isinstance(value, str):
                    try:
                        exif_dict[ifd][tag_id] = base64.b64decode(value)
                    except Exception:
                        exif_dict[ifd][tag_id] = value.encode("utf-8", errors="replace")
                elif isinstance(value, bytes):
                    exif_dict[ifd][tag_id] = value
                else:
                    exif_dict[ifd][tag_id] = str(value).encode("utf-8", errors="replace")
            elif tag_type == 2:  # ASCII
                if isinstance(value, str):
                    exif_dict[ifd][tag_id] = value.encode("utf-8", errors="replace")
                elif isinstance(value, bytes):
                    exif_dict[ifd][tag_id] = value
                else:
                    exif_dict[ifd][tag_id] = str(value).encode("utf-8", errors="replace")
            else:
                exif_dict[ifd][tag_id] = value
    return exif_dict


class AET_ImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {"default": "", "image_upload": True, "tooltip": "Drag & drop an image or click to browse the input folder."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_path")
    FUNCTION = "load"
    CATEGORY = "image/loaders"

    def load(self, image):
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, image)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return (tensor, file_path)


class AET_ReferenceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {"default": "", "image_upload": True, "tooltip": "Drag & drop a reference photo taken with a real camera or scanner."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("ref_image", "ref_file_path", "ref_EXIF_json")
    FUNCTION = "load"
    CATEGORY = "image/loaders"

    def load(self, image):
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, image)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        exif_json = ""
        try:
            exif_dict = piexif.load(file_path)
            exif_json = _exif_dict_to_json(exif_dict)
        except Exception as e:
            exif_json = json.dumps({"error": str(e)})

        return (tensor, file_path, exif_json)


class AET_PresetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_model": (_list_cameras(), {"tooltip": "Select a camera preset from presets/cameras/ or presets/custom/."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("exif_json",)
    FUNCTION = "load"
    CATEGORY = "image/loaders"

    def load(self, camera_model):
        if camera_model == "None":
            return (json.dumps({"error": "No camera preset selected"}),)

        preset = _load_camera(camera_model)
        if not preset:
            return (json.dumps({"error": f"Preset not found: {camera_model}"}),)

        if "exif" in preset:
            return (json.dumps(preset["exif"], ensure_ascii=False),)

        exif_dict = _camera_to_exif(preset)
        return (_exif_dict_to_json(exif_dict),)


class AET_PresetSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_file_path": ("STRING", {"forceInput": True, "tooltip": "Path to the reference image (from AET_ReferenceLoader)."}),
                "preset_name": ("STRING", {"default": "", "tooltip": "Name of the preset file (without .json)."}),
                "save_exif": ("BOOLEAN", {"default": False, "tooltip": "Save EXIF tags from the reference image into the preset. Debug always shows the EXIF first, so you can decide."}),
                "save_profile": ("BOOLEAN", {"default": False, "tooltip": "Save pixel profile (not implemented yet)."}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "If False, refuse to overwrite an existing preset file."}),
                "artist": ("STRING", {"default": "", "tooltip": "Optional. If non-empty, written into EXIF Artist field of the preset. If empty, the reference value is kept."}),
                "copyright": ("STRING", {"default": "", "tooltip": "Optional. If non-empty, written into EXIF Copyright field of the preset. If empty, the reference value is kept."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
    FUNCTION = "save"
    CATEGORY = "image/save"
    OUTPUT_NODE = True

    def save(self, ref_file_path, preset_name, save_exif, save_profile, overwrite, artist="", copyright=""):
        debug_lines = []

        if not ref_file_path or not ref_file_path.strip() or not os.path.exists(ref_file_path):
            debug_lines.append(f"Error: reference file not found: {ref_file_path}")
            return ("\n".join(debug_lines),)

        exif_serializable = None
        try:
            exif_dict = piexif.load(ref_file_path)
            exif_serializable = json.loads(_exif_dict_to_json(exif_dict))
            debug_lines.append("EXIF from reference:")
            for ifd in ("0th", "Exif", "GPS", "1st"):
                for tag_name, value in exif_serializable.get(ifd, {}).items():
                    debug_lines.append(f"  {ifd}/{tag_name}: {value}")
        except Exception as e:
            debug_lines.append(f"Error reading EXIF: {e}")

        if not save_exif and not save_profile:
            debug_lines.append("Nothing to save: both save_exif and save_profile are False.")
            return ("\n".join(debug_lines),)

        if not preset_name or not preset_name.strip():
            debug_lines.append("Error: preset_name is empty.")
            return ("\n".join(debug_lines),)

        preset = {"display_name": preset_name.strip()}

        if save_exif and exif_serializable is not None:
            preset["exif"] = exif_serializable
            if artist:
                preset["exif"]["0th"]["Artist"] = artist
            if copyright:
                preset["exif"]["0th"]["Copyright"] = copyright

        if save_profile:
            debug_lines.append("Profile saving is not implemented yet. Skipped.")

        _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _CUSTOM_DIR / f"{preset_name.strip()}.json"

        if out_path.exists() and not overwrite:
            debug_lines.append(f"Error: preset already exists: {out_path}. Set overwrite=True to replace.")
            return ("\n".join(debug_lines),)

        try:
            out_path.write_text(json.dumps(preset, ensure_ascii=False, indent=2), encoding="utf-8")
            debug_lines.append(f"Saved preset: {out_path}")
        except Exception as e:
            debug_lines.append(f"Error writing preset: {e}")

        return ("\n".join(debug_lines),)


class AET_PixelEmulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scanner_model": (_list_scanners(), {"tooltip": "Select a scanner preset. [Film] = film scanner, [Flatbed] = document/flatbed scanner."}),
                "frequency_cutoff": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.4, "step": 0.01, "tooltip": "FFT cutoff radius. Lower = more high frequencies affected."}),
                "grain_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Strength of synthetic grain added to high frequencies."}),
                "row_shift": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sub-pixel horizontal shift per row (scanner head jitter). 0 = disabled."}),
                "lab_chroma_strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5, "tooltip": "LAB a/b chroma noise strength. 0 = disabled."}),
                "enable_pixel_processing": ("BOOLEAN", {"default": True, "tooltip": "If False, pixels are passed through unchanged (EXIF is still generated)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Random seed for grain generation."}),
            },
            "optional": {
                "file_path": ("STRING", {"forceInput": True, "tooltip": "Optional: path to the input image (for EXIF debug). Connect from AET_ImageLoader."}),
                "ref_file_path": ("STRING", {"forceInput": True, "tooltip": "Optional: path to a reference image. If connected, its EXIF is used instead of the scanner preset."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "debug_exif", "exif_json")
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def process(self, image, scanner_model, frequency_cutoff, grain_amount, row_shift, lab_chroma_strength, enable_pixel_processing, seed, file_path="", ref_file_path=""):
        debug_lines = []
        exif_dict = None

        if ref_file_path and ref_file_path.strip():
            debug_lines.append(f"ref_file_path connected: {ref_file_path} -> scanner preset ignored.")
            if os.path.exists(ref_file_path):
                try:
                    exif_dict = piexif.load(ref_file_path)
                    debug_lines.append("EXIF from reference:")
                    for ifd in ("0th", "Exif", "GPS", "1st"):
                        for tag, value in exif_dict.get(ifd, {}).items():
                            tag_name = piexif.TAGS[ifd][tag]["name"]
                            debug_lines.append(f"  {ifd}/{tag_name}: {value}")
                except Exception as e:
                    debug_lines.append(f"Failed to read reference EXIF: {e}")
            else:
                debug_lines.append(f"Reference file not found: {ref_file_path}")
        else:
            if scanner_model == "None":
                debug_lines.append("scanner_model = None -> no EXIF will be generated.")
            else:
                preset = _load_scanner(scanner_model)
                if preset:
                    exif_dict = _scanner_to_exif(preset)
                    debug_lines.append(f"Preset: {scanner_model} -> generated EXIF:")
                    for ifd in ("0th", "Exif"):
                        for tag, value in exif_dict.get(ifd, {}).items():
                            tag_name = piexif.TAGS[ifd][tag]["name"]
                            debug_lines.append(f"  {ifd}/{tag_name}: {value}")
                else:
                    debug_lines.append(f"Preset not found: {scanner_model}")

        if file_path and file_path.strip():
            debug_lines.append(f"file_path connected: {file_path}")

        debug_str = "\n".join(debug_lines)
        print(f"[AET_PixelEmulator] {debug_str}")

        exif_json = ""
        if exif_dict:
            exif_json = _exif_dict_to_json(exif_dict)

        if enable_pixel_processing:
            processed_tensor = apply_film_scan_processing(
                image, frequency_cutoff, grain_amount, seed,
                row_shift=row_shift, lab_chroma_strength=lab_chroma_strength
            )
        else:
            debug_str += "\nPixel processing disabled: passing image through unchanged."
            processed_tensor = image

        return (processed_tensor, debug_str, exif_json)


class AET_ImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exif_json": ("STRING", {"forceInput": True, "tooltip": "Required. JSON with EXIF tags from AET_PixelEmulator or AET_PresetLoader. Saving is refused if empty or invalid."}),
                "filename_prefix": ("STRING", {"default": "SCAN", "tooltip": "Prefix for the output filename."}),
                "output_subfolder": ("STRING", {"default": "scans", "tooltip": "Subfolder inside ComfyUI/output/. Nested paths like scans/2026 are allowed."}),
                "quality": ("INT", {"default": 88, "min": 1, "max": 100, "tooltip": "JPEG quality (1-100). Higher = better quality, larger file."}),
                "subsampling": (["4:4:4", "4:2:2", "4:2:0"], {"default": "4:4:4", "tooltip": "JPEG chroma subsampling. 4:4:4 keeps the most detail (best for film grain)."}),
                "date_override": ("STRING", {"default": "", "tooltip": "Optional. Format: YYYY:MM:DD HH:MM:SS. If empty, current date/time is used."}),
                "overwrite_artist_copyright": ("BOOLEAN", {"default": False, "tooltip": "If True, overwrite Artist and Copyright in EXIF with the values below (only if non-empty)."}),
                "artist": ("STRING", {"default": "", "tooltip": "Optional. Used only if overwrite_artist_copyright is True and this field is non-empty."}),
                "copyright": ("STRING", {"default": "", "tooltip": "Optional. Used only if overwrite_artist_copyright is True and this field is non-empty."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "image/save"
    OUTPUT_NODE = True

    def save(self, image, exif_json, filename_prefix, output_subfolder, quality, subsampling, date_override, overwrite_artist_copyright=False, artist="", copyright=""):
        if not exif_json or not exif_json.strip():
            raise ValueError("exif_json is required. Refusing to save without EXIF.")

        try:
            exif_serializable = json.loads(exif_json)
        except Exception as e:
            raise ValueError(f"Invalid exif_json: {e}")

        if "error" in exif_serializable:
            raise ValueError(f"exif_json contains error: {exif_serializable['error']}")

        # Визначаємо дату: override або поточна
        if date_override and date_override.strip():
            date_str = date_override.strip()
            try:
                datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Invalid date_override format: {date_str}. Expected YYYY:MM:DD HH:MM:SS")
        else:
            date_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

# Замінюємо дату в EXIF перед записом
        for ifd in ("0th", "Exif"):
            for tag_name in ("DateTime", "DateTimeOriginal", "DateTimeDigitized"):
                if tag_name in exif_serializable.get(ifd, {}):
                    exif_serializable[ifd][tag_name] = date_str

        # Скидаємо Orientation, бо пікселі вже орієнтовані правильно
        if "Orientation" in exif_serializable.get("0th", {}):
            exif_serializable["0th"]["Orientation"] = 1

        if overwrite_artist_copyright:
            if artist:
                exif_serializable["0th"]["Artist"] = artist
            if copyright:
                exif_serializable["0th"]["Copyright"] = copyright

        if "Exif" not in exif_serializable:
            exif_serializable["Exif"] = {}
        exif_serializable["Exif"]["PixelXDimension"] = image.shape[2]
        exif_serializable["Exif"]["PixelYDimension"] = image.shape[1]

        exif_dict = _json_to_exif_dict(exif_serializable)

        try:
            exif_bytes = piexif.dump(exif_dict)
        except Exception as e:
            raise ValueError(f"Failed to build EXIF bytes: {e}")

        output_dir = Path(folder_paths.get_output_directory()) / output_subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        file_date = datetime.now().strftime("%Y%m%d")
        counter = 1
        while True:
            filename = f"{filename_prefix}{file_date}_{counter:02d}.jpg"
            full_path = output_dir / filename
            if not full_path.exists():
                break
            counter += 1

        subsampling_map = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
        subsampling_value = subsampling_map.get(subsampling, 0)

        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            save_path = full_path if i == 0 else full_path.with_name(f"{full_path.stem}_{i+1}.jpg")
            img.save(save_path, "JPEG", quality=quality, subsampling=subsampling_value, exif=exif_bytes)

        return (str(full_path),)


NODE_CLASS_MAPPINGS = {
    "AET_ImageLoader": AET_ImageLoader,
    "AET_ReferenceLoader": AET_ReferenceLoader,
    "AET_PresetLoader": AET_PresetLoader,
    "AET_PresetSaver": AET_PresetSaver,
    "AET_PixelEmulator": AET_PixelEmulator,
    "AET_ImageSaver": AET_ImageSaver,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AET_ImageLoader": "AET Image Loader",
    "AET_ReferenceLoader": "AET Reference Loader",
    "AET_PresetLoader": "AET Preset Loader",
    "AET_PresetSaver": "AET Preset Saver",
    "AET_PixelEmulator": "AET Pixel Emulator",
    "AET_ImageSaver": "AET Image Saver",
}