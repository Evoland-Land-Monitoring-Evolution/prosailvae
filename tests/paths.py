from pathlib import Path

ROOT_DIR = Path("/work/CESBIO/projects/DeepChange/PROSAILVAE")
TMP_DIR = Path("/tmp")
if Path("/usr/local/stok/DATA/MMDC").exists():
    ROOT_DIR = Path("/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE")
elif Path("/gpfsscratch/rech/adz/uzh16pa/ProsailVAE").exists():
    ROOT_DIR = Path("/gpfsscratch/rech/adz/uzh16pa/ProsailVAE/PROSAILVAE")
    TMP_DIR = Path("/gpfsscratch/rech/adz/uzh16pa/tmp")
PATCHES_DIR = Path(f"{ROOT_DIR}/s2_patch_dataset/")
