
Semi-Supervised Annotation with SAM2 (PyQt5)
============================================

An interactive annotation tool built on **Segment Anything 2 (SAM2)** with a clean **PyQt5** interface. Annotate with hover + click-refine, assign classes, save **YOLO segmentation** labels, and **propagate** those labels through an entire image folder. Includes a blocking progress dialog during propagation and smart merging so multiple runs _append_ labels without overwriting.

This app uses the SAM2 backend from: https://github.com/Gy920/segment-anything-2-real-time

* * *

✨ Features
----------

*   🖱️ **Hover** segmentation preview + **click refine** (add/subtract points).
*   🏷️ Sidebar **class list** (customizable via `--classes`).
*   💾 Save annotations as **YOLO-seg polygons** (`.txt` per image).
*   📂 **Folder browser** (loads images on demand to avoid RAM spikes).
*   🔁 **Propagation** using SAM2’s camera/video propagator (seeded by your clicks).
*   ➕ **Merges** new propagation results into existing `.txt` files (no overwrite).
*   ⏳ **Blocking progress dialog** with per-image status during propagation.

* * *

🎬 Demo
-------

▶️ **YouTube:** [SAM2 Interactive & Propagation UI](https://www.youtube.com/watch?v=XXXXXXXXXXX) (Replace with your actual demo link.)

* * *

📁 Repository Layout
--------------------

    Semi_supervised_annotation_with_SAM2/
    ├─ run.py                 # Main PyQt5 app
    ├─ src/                       # Helpers (ObjRecord, Click, etc.)
    ├─ models/                    # (Populated by ./download.sh)
    ├─ images/                    # Your image folder(s)
    ├─ requirements.txt
    ├─ download.sh                # Downloads SAM2 checkpoints and sam2 repo
    └─ README.md
    

* * *

⚙️ Setup
--------

### 1) Clone this project

    git clone https://github.com/NomanAnjum09/Semi_supervised_annotation_with_SAM2
    cd Semi_supervised_annotation_with_SAM2
    
### 2) Download checkpoints and SAM repo

    ./download.sh

### 3) Create & activate a virtual environment

**Linux / macOS (bash/zsh)**

    python3 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    

**Windows (PowerShell)**

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    

### 4) Install the real-time SAM2 dependency (editable)

    cd segment-anything-2-real-time/
    pip install -e .
    cd ..
    

### 5) Install project requirements

    pip install -r requirements.txt
    


* * *

🚀 Running the App
------------------

Annotate images in a folder and propagate:

    python sam2_ui.py \
      --images path/to/your/images_folder \
      --model_type large \
      --classes Person Car Tree
    

### CLI Arguments

*   `--images` : Folder containing images (PNG/JPG/BMP/TIFF/WEBP).
*   `--model_type` : `tiny` | `small` | `large` — controls which checkpoint is used (downloaded via `download.sh`).
*   `--classes` : Space-separated list of class names shown in the sidebar.

* * *

🖱️ How to Annotate
-------------------

*   Hover over the canvas → live SAM2 mask preview.
*   **Left-click** = Add point (positive); **Right-click** = Subtract point (negative).
*   Click a class in the sidebar to **save** the current object under that class.
*   Click a saved (frozen) object to **edit** it; press **Delete** to remove it.

### Keyboard Shortcuts

*   **Z** → Undo last click
*   **C** → Clear clicks for current (pending) object
*   **Delete** → Delete the object being edited
*   **Esc** → Cancel editing / clear pending

### Images Panel

The right panel lists images from the given folder. Click a filename to load that image on demand (keeps memory usage low). If a matching `.txt` already exists, polygons are loaded and rendered immediately.

* * *

🔁 Propagation (Semi-Supervised)
--------------------------------

1.  Open a frame, add clicks, and **save** one or more objects.
2.  Press **Propagate**.
3.  The app initializes SAM2’s camera propagator with this frame and your saved click prompts, then iterates the entire folder.
4.  A **blocking progress dialog** appears, showing per-image status until completion.

### Merging Behavior

*   Per-image labels are written to `<image_basename>.txt` alongside each image.
*   Subsequent propagations **append/merge** new polygons; existing ones are preserved (exact duplicates are de-duplicated).
*   Example: Run once with two objects; run again later for a third object → the second run appends only the new object’s polygons.

* * *

📦 YOLO Segmentation Format
---------------------------

For an image `img001.jpg`, a file `img001.txt` is created:

    0 0.123456 0.234567 0.223456 0.234567 0.300000 0.400000 ...
    2 0.543210 0.678910 0.600000 0.700000 ...
    

*   Each line = one polygon/object.
*   `class_id` followed by normalized `x1 y1 x2 y2 ... xN yN`.
*   Multiple polygons per file are supported.

* * *

⚡ Performance Tips
------------------

*   Use **CUDA** if available (`torch.cuda.is_available()`) for faster propagation.
*   The app uses autocast (`bfloat16`) on CUDA for speed.
*   Very large images will slow down propagation; consider downscaling if needed.

* * *

# Troubleshooting
- Qt “xcb” plugin errors on Linux
    Use headless OpenCV and install system libs:
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless numpy
    pip install --no-cache-dir "numpy>=2.0" "opencv-python-headless>=4.8.1.78"


    sudo apt-get install -y libxcb1 libx11-xcb1 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-randr0 libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxcb-xinerama0 \
    libglu1-mesa libopengl0

* * *

🗺️ Roadmap
-----------

*   Polygon vertex editing (drag to adjust).
*   Export to COCO JSON / Pascal VOC.
*   Selective propagation (by class / checkbox list).
*   Session autosave & undo/redo across sessions.

* * *

🙏 Acknowledgements
-------------------

*   Meta AI — Segment Anything / SAM2
*   PyQt5, OpenCV, NumPy, PyTorch

* * *

📄 License
----------

MIT