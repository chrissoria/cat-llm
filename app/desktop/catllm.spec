# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for CatLLM Desktop."""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

DESKTOP_DIR = os.path.dirname(os.path.abspath(SPEC))
APP_DIR = os.path.dirname(DESKTOP_DIR)
ROOT_DIR = os.path.dirname(APP_DIR)

# Collect Streamlit data files (templates, static assets) + metadata
streamlit_datas = collect_data_files("streamlit")

# Packages that check their own version via importlib.metadata
metadata_packages = [
    "streamlit", "altair", "pandas", "numpy", "matplotlib",
    "Pillow", "requests", "cat-llm", "cat-stack",
]
metadata_datas = []
for pkg in metadata_packages:
    try:
        metadata_datas += copy_metadata(pkg)
    except Exception:
        pass

# Collect all catllm-ecosystem hidden imports
hidden_imports = (
    collect_submodules("catllm")
    + collect_submodules("cat_stack")
    + collect_submodules("cat_survey")
    + collect_submodules("catvader")
    + collect_submodules("catademic")
    + collect_submodules("cat_cog")
    + collect_submodules("streamlit")
    + [
        "pandas",
        "openpyxl",
        "requests",
        "regex",
        "reportlab",
        "matplotlib",
        "PIL",
        "fitz",        # PyMuPDF
        "pymupdf",
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "webview",
        "webview.platforms.cocoa",
        "objc",
        "Foundation",
        "AppKit",
        "WebKit",
    ]
)

a = Analysis(
    [os.path.join(DESKTOP_DIR, "launcher.py")],
    pathex=[APP_DIR, ROOT_DIR],
    datas=[
        # App source files — land at Frameworks root so launcher finds them
        (os.path.join(APP_DIR, "main.py"), "."),
        (os.path.join(APP_DIR, "config.py"), "."),
        (os.path.join(APP_DIR, "session.py"), "."),
        (os.path.join(APP_DIR, "components"), "components"),
        (os.path.join(APP_DIR, "domains"), "domains"),
        (os.path.join(APP_DIR, "functions"), "functions"),
        (os.path.join(APP_DIR, "assets"), "assets"),
        (os.path.join(APP_DIR, ".streamlit"), ".streamlit"),
    ]
    + streamlit_datas
    + metadata_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Standard-library modules we never touch
        "tkinter",
        "unittest",
        "xmlrpc",
        "test",
        "lib2to3",
        "pydoc_data",
        # Heavy ML/scientific deps not needed for API-based classification
        "torch",
        "torchvision",
        "torchaudio",
        "accelerate",
        "transformers",
        "safetensors",
        "tensorflow",
        "jax",
        "flax",
        "playwright",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "scipy",
        "sklearn",
        "scikit-learn",
        "skimage",
        "scikit-image",
        "spacy",
        "astropy",
        "statsmodels",
        "h5py",
        "nltk",
        "sympy",
        "IPython",
        "ipykernel",
        "ipywidgets",
        "notebook",
        "jupyterlab",
        "jupyter",
        "jupyter_client",
        "jupyter_core",
        "qtconsole",
        "pytest",
        # Plotting backends we never load — bokeh alone is ~100 MB
        "bokeh",
        "plotly",
        "seaborn",
        # NOTE: keep altair — Streamlit imports it eagerly during startup,
        # excluding it causes "ModuleNotFoundError: altair" at first render.
        "vega_datasets",
        # nbconvert pulls in pandoc-like tooling we don't need
        "nbconvert",
        "nbformat",
        # Selenium / webdriver / requests-html
        "selenium",
        "lxml",
        "html5lib",
        "bs4.builder._lxml",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CatLLM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # No terminal window
    target_arch=None,        # Build for current architecture
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CatLLM",
)

app = BUNDLE(
    coll,
    name="CatLLM.app",
    icon=os.path.join(DESKTOP_DIR, "icon.icns"),
    bundle_identifier="com.catllm.desktop",
    info_plist={
        "CFBundleName": "CatLLM",
        "CFBundleDisplayName": "CatLLM",
        "CFBundleShortVersionString": "3.1.1",
        "CFBundleVersion": "3.1.1",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
