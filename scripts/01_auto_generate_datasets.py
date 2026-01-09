import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
outputs_dir = os.path.join(project_root, "data", "01_dataset")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import tqdm as tqdm
import random
import pickle
import time
import torch
from input_utils import *
from topology import *

SAVE_EVERY = 50
MAX_TIME = 60
VOLFRAC = 0.3
PENAL = 1.5
RMIN = 1.5
AUTO_LOADS = True

def main():
    while True:
        version_nr = get_valid_input("Enter version number (e.g., 01, 02, etc.):\n", int, "Invalid version number. Please enter a numeric value.")
        nr_samples = get_valid_input("Enter number of samples to be generated (suggestion: 1000):\n", int, "Invalid number of samples. Please enter a numeric value.")
        break

    while True:
        nelx = get_valid_input("Enter number of horizontal elements (suggestion: 40):\n", int, "Invalid input. Please enter a numeric value.")
        nely = get_valid_input("Enter number of vertical elements (suggestion: 20):\n", int, "Invalid input. Please enter a numeric value.")
        print(f"Valid entries, nelx = {nelx}, nely = {nely}")
        break

    # Preallocate arrays
    X_data = np.zeros((nr_samples, nely+1, nelx+1, 6), dtype=np.float32)
    Y_data = np.zeros((nr_samples, nely, nelx), dtype=np.float32)
    ndof = 2 * (nelx + 1) * (nely + 1)

    filename = f"topopt_dataset_v{version_nr:02d}_{nr_samples}samples.pt"
    save_path = os.path.join(outputs_dir, filename)
    print(f"\nStarting generation of {nr_samples} samples...")

    for i in tqdm(range(nr_samples)):
        # Start model
        simp = SIMPOptimizer(nelx,nely,VOLFRAC,PENAL,RMIN)
        load = sp.lil_matrix((ndof,1))
        nr_loads = random.randint(3,7)

        for _ in range(nr_loads):
            dof_idx = random.randrange(ndof)
            load[dof_idx, 0] = random.choice([-1, 1])

        # Calculate initial state
        U_init = simp._solve_fem(load)

        # Compute strains
        epsilon_x, epsilon_y, gamma_xy = simp.compute_strains(U_init)
        # Upsample strains to nodes (Element -> Node)
        epsilon_x_node = SIMPOptimizer.upsample_strain_to_nodes(epsilon_x)
        epsilon_y_node = SIMPOptimizer.upsample_strain_to_nodes(epsilon_y)
        gamma_xy_node = SIMPOptimizer.upsample_strain_to_nodes(gamma_xy)

        # Volume fraction channel
        vol_channel = np.ones((nely + 1, nelx + 1)) * VOLFRAC

        # Process displacements (Vector -> Grid)
        U_x = U_init[0::2, 0]
        U_y = U_init[1::2, 0]
        U_x_grid = U_x.reshape((nelx + 1, nely + 1)).T
        U_y_grid = U_y.reshape((nelx + 1, nely + 1)).T

        # Run optimization (Target Label)
        # optimize() updates simp.x internally to the final structure
        x_final, _ = simp.optimize(load, max_time=MAX_TIME, plot=False)

        if x_final is None:
            print(f"Sample {i} skipped due to timeout.\n")
            continue

        # Store data
        # Input: Stack 6 channels [Ux, Uy, Ex, Ey, Gxy, Vol]
        # Axis -1 puts channels at the end: (H, W, C)
        X_data[i] = np.stack([
            U_x_grid, 
            U_y_grid, 
            epsilon_x_node, 
            epsilon_y_node, 
            gamma_xy_node, 
            vol_channel
        ], axis=-1)
        # Output: Final density distribution
        Y_data[i] = x_final

        # Checkpoint safeguard (saving)
        if (i + 1) % SAVE_EVERY == 0:
            torch.save({
                'X': torch.tensor(X_data[:i+1]), 
                'y': torch.tensor(Y_data[:i+1])
            }, save_path)
    
    # Final save
    torch.save({
        'X': torch.tensor(X_data), 
        'y': torch.tensor(Y_data)
    }, save_path)

    print(f"\nDataset generation complete. Saved to: {save_path}")

if __name__ == "__main__":
    main()