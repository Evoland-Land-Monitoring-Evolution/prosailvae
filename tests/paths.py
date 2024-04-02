from pathlib import Path

ROOT_DIR = Path("/work/CESBIO/projects/DeepChange/PROSAILVAE")
TMP_DIR = Path("/tmp")
if Path("/usr/local/stok/DATA/MMDC").exists():
    ROOT_DIR = Path("/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE")
elif Path("/gpfsscratch/rech/adz/uzh16pa/ProsailVAE").exists():
    ROOT_DIR = Path("/gpfsscratch/rech/adz/uzh16pa/ProsailVAE/PROSAILVAE")
    TMP_DIR = Path("/gpfsscratch/rech/adz/uzh16pa/tmp")
PATCHES_DIR = Path(f"{ROOT_DIR}/s2_patch_dataset/")

MMDC_DATASET_DIR = "/work/CESBIO/projects/DeepChange/Ekaterina/MMDC_OE/"
if Path("/gpfsscratch/rech/adz/uzh16pa/MMDC").exists():
    MMDC_DATASET_DIR = "/gpfsscratch/rech/adz/uzh16pa/MMDC/"
MMDC_TILES_CONFIG_DIR = f"{MMDC_DATASET_DIR}/tiles_conf_training/tiny"

MMDC_DATA_COMPONENTS = 8
