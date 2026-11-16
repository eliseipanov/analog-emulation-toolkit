import sys

_missing = []
try:
    import numpy
except ImportError:
    _missing.append("numpy")
try:
    import scipy
except ImportError:
    _missing.append("scipy")
try:
    import piexif
except ImportError:
    _missing.append("piexif")

if _missing:
    print("\n" + "="*60)
    print("  Film Scan Emulator: missing required Python packages!")
    print(f"  Missing: {', '.join(_missing)}")
    print("  Run this in the Python environment ComfyUI uses:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    print("  (or reinstall the node via ComfyUI Manager, which handles dependencies)")
    print("="*60 + "\n")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./web"

