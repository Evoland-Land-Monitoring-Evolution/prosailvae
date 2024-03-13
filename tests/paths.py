from pathlib import Path

ROOT_DIR = "/work/CESBIO/projects/DeepChange/PROSAILVAE"
if Path("/usr/local/stok/DATA/MMDC").exists():
    ROOT_DIR = "/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE"
elif Path("/gpfsscratch/rech/adz/uzh16pa/ProsailVAE").exists():
    ROOT_DIR = "/gpfsscratch/rech/adz/uzh16pa/ProsailVAE/PROSAILVAE"
PATCHES_DIR = Path(f"{ROOT_DIR}/s2_patch_dataset/")
