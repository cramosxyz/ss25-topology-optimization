import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
data_dir = os.path.join(project_root, "data", "01_raw")
outputs_dir = os.path.join(project_root, "data", "02_ct_volumes")

import cv2
import glob
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ct_utils import *
from ct_analysis import *
from skimage import io, transform
from skimage.transform import radon, iradon
from sklearn.metrics import mean_squared_error
from scipy.ndimage import median_filter, gaussian_filter

SIZE = (256,256)

def main():
    while True:
        measurement_day = input("Enter measurement day in yyyy-mm-dd format:\n").strip()
        if not measurement_day:
            print("Input cannot be empty. Please try again.")
            continue
        
        day_path = os.path.join(data_dir, measurement_day)
        if not os.path.isdir(day_path):
            print(f"Folder not found: {day_path}")
            print(f"Available folders in data/: {os.listdir(data_dir)}")
            continue
        break
    
    while True:
        scan_nr = input("Enter scan folder name (e.g., scan01):\n").strip()
        if not scan_nr:
            print("Input cannot be empty. Please try again.")
            continue

        scan_path = os.path.join(day_path, scan_nr)
        if not os.path.isdir(scan_path):
            print(f"Scan folder not found: {scan_path}")
            if os.path.exists(day_path):
                print(f"Available scans: {os.listdir(day_path)}")
            continue

        config_path = os.path.join(scan_path, "0_config.json")
        if not os.path.isfile(config_path):
            print(f"\nMissing Config File: {config_path}")
            print("Performing assessment automatically, for better results use notebook 01-CT_Reco.ipynb first.")

        break
    print(f"Valid data found in: {measurement_day} / {scan_nr}")

    while True:
        modality = input("\nEnter reco modality (e.g., xr, op, all):\n").strip()
        if modality in ["op", "xr", "all"]:
            break
        else:
            print("Incorrect modality. Only 'xr', 'op' or 'all' accepted")

    # Load data
    xray_path = os.path.join(scan_path)
    optical_path = os.path.join(scan_path)

    if modality in ["xr", "all"]:
        print("\nLoading xray data")
        xr_empty_stack = load_image_stack(os.path.join(day_path, 'empty'), "*.TIFF")
        xr_proj_stack = load_image_stack(xray_path, "*.TIFF")

        xray_processed = compute_flat_field_correction(xr_proj_stack, xr_empty_stack, dark=None)

    if modality in ["op", "all"]:
        print("\nLoading optical data")
        op_empty_stack = load_image_stack(os.path.join(day_path,'empty'), "*.npy", SIZE, convert_gray=True, normalize_factor=255)
        op_dark_stack = load_image_stack(os.path.join(day_path,'dark'), "*.npy", SIZE, convert_gray=True, normalize_factor=255)
        op_proj_stack = load_image_stack(optical_path, "*.npy", SIZE, convert_gray=True, normalize_factor=255)

        optical_processed = compute_flat_field_correction(op_proj_stack, op_empty_stack, op_dark_stack)

    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            params = json.load(f)
    else:
        if modality in ["xr", "all"]:
            y_edge, x_edge, x_edge2 = find_cropping_edges(xr_proj_stack)
        else:
            y_edge, x_edge, x_edge2 = None, None, None
        
        params = {
            "description": "Automated Assessment",
            "optical": {
                "v_min": None,
                "v_max": None,
                "h_min": None,
                "h_max": None
            },
            "xray": {
                "v_min": x_edge,
                "v_max": x_edge2,
                "h_min": y_edge,
                "h_max": None
            }
        }
    
    fullangle_projection = None

    if modality in ["xr", "all"]:
        del xr_empty_stack, xr_proj_stack

        # Reconstructing X-RAY
        print("\n-- X-ray CT --")

        _, fullangle_projection = reconstruct_volume(
            data_stack=xray_processed,
            params=params['xray'],
            output_dir=os.path.join(outputs_dir, measurement_day, "xray"),
            file_prefix=scan_nr,
            angle_step=1.8,
            fullangle_projection=fullangle_projection
        )

    if modality in ["op", "all"]:
        del op_empty_stack, op_dark_stack, op_proj_stack

        # Reconstructing OPTICAL
        print("\n-- Optical CT --")

        reconstruct_volume(
            data_stack=optical_processed,
            params=params['optical'],
            output_dir=os.path.join(outputs_dir, measurement_day, "optical"),
            file_prefix=scan_nr,
            angle_step=1.8,
            rotate=True,
            fullangle_projection=fullangle_projection
        )

    print("\nReconstructions complete!")

if __name__ == "__main__":
    main()