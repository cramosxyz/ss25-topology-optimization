#%% [markdown]
# # Topology Optimization Data Creation
# Imports and environment setup

#%%
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import pickle
import time

#%% [markdown]
# ## SIMP functions

#%%
def lk():
    E = 1.0
    nu = 0.3
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    KE = E / (1 - nu**2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                     [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                     [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                     [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                     [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                     [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                     [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                     [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE

def FE(nelx, nely, x, penal, F):
    KE = lk()
    ndof = 2 * (nelx+1) * (nely+1)
    K = sp.lil_matrix((ndof, ndof))
    # F = sp.lil_matrix((ndof,1))
    U = np.zeros((ndof,1))
    
    # Assemble K
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
            K[np.ix_(edof, edof)] += x[ely, elx]**penal * KE
            
    K = K.tocsr()
    
    # # Load: downward in the middle-right
    # F[2*((nely+1)*nelx + nely//2)+1,0] = -1

    # ## TESTING
    # # Info: F[... +1 at the end] -> Y DOF
    # # It counts from down-up, then left to right
    # # Fy = -1 downwards, +1 upwards
    # # Fx = -1 left, +1 right
    # F[2 * ((nely + 1) * (nelx) + (nely // 2)), 0] = 1  # Far right, center, right
    # F[2 * ((nely + 1) * (nelx//2)) + 1, 0] = -1  # Middle, down, downwards
    # F[2 * ((nely + 1) * (nelx//2) + nely) + 1, 0] = 1  # Middle, up, upwards
    
    # Fixities: left edge
    fixeddofs = np.arange(0, 2*(nely+1), 1)
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    
    # Solve
    U[freedofs,0] = spla.spsolve(K[freedofs,:][:,freedofs], F[freedofs].toarray().flatten())
    U[fixeddofs,0] = 0
    return U

def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    for i in range(nelx):
        for j in range(nely):
            s = 0
            for k in range(max(i-int(rmin),0), min(i+int(rmin)+1, nelx)):
                for l in range(max(j-int(rmin),0), min(j+int(rmin)+1, nely)):
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    s += max(0, fac)
                    dcn[j,i] += max(0, fac) * x[l,k] * dc[l,k]
            dcn[j,i] /= (x[j,i]*s)
    return dcn

def OC(nelx, nely, x, volfrac, dc):
    l1, l2 = 0, 1e5
    move = 0.2
    while (l2-l1) > 1e-4:
        lmid = 0.5*(l2+l1)
        xnew = np.maximum(0.001, np.maximum(x-move, np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/lmid)))))
        if np.sum(xnew) - volfrac*nelx*nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew

def compute_strains(nelx, nely, U):
    """
    Compute element-wise strains from nodal displacements.
    Returns epsilon_x, epsilon_y, gamma_xy arrays of shape (nely, nelx)
    """
    U_x = U[0::2, 0]  # every even DOF -> x displacement
    U_y = U[1::2, 0]  # every odd DOF -> y displacement

    U_x_grid = U_x.reshape((nelx+1, nely+1)).T
    U_y_grid = U_y.reshape((nelx+1, nely+1)).T

    epsilon_x = np.zeros((nely, nelx))
    epsilon_y = np.zeros((nely, nelx))
    gamma_xy = np.zeros((nely, nelx))

    for elx in range(nelx):
        for ely in range(nely):
            # node indices
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            n3 = (nely+1)*(elx+1) + (ely+1)
            n4 = (nely+1)*elx + (ely+1)

            u_e = np.array([U_x[n1], U_y[n1],
                            U_x[n2], U_y[n2],
                            U_x[n3], U_y[n3],
                            U_x[n4], U_y[n4]])

            # finite difference approx
            epsilon_x[ely, elx] = 0.5*((u_e[2] - u_e[0]) + (u_e[4] - u_e[6]))
            epsilon_y[ely, elx] = 0.5*((u_e[5] - u_e[1]) + (u_e[7] - u_e[3]))
            gamma_xy[ely, elx] = 0.5*((u_e[3] - u_e[1]) + (u_e[2] - u_e[0]))

    return epsilon_x, epsilon_y, gamma_xy

def upsample_strain_to_nodes(strain_element):
    nely, nelx = strain_element.shape
    strain_node = np.zeros((nely+1, nelx+1))
    # top-left contribution
    strain_node[:-1, :-1] += strain_element
    # top-right
    strain_node[:-1, 1:] += strain_element
    # bottom-left
    strain_node[1:, :-1] += strain_element
    # bottom-right
    strain_node[1:, 1:] += strain_element
    strain_node /= 4
    return strain_node

def top(nelx, nely, volfrac, penal, rmin, F, plot=False, max_time=60):
    start_time = time.time()
    x = np.ones((nely, nelx))*volfrac
    change = 1.0
    while change > 0.01:
        # Abort if taking too long
        if time.time() - start_time > max_time:
            print("\nSIMP aborted: exceeded max_time")
            return x  # or np.zeros_like(x) if you prefer

        xold = x.copy()
        U = FE(nelx, nely, x, penal, F)
        KE = lk()
        dc = np.zeros((nely, nelx))
        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely+1)*elx + ely
                n2 = (nely+1)*(elx+1) + ely
                Ue = U[[2*n1,2*n1+1,2*n2,2*n2+1,2*n2+2,2*n2+3,2*n1+2,2*n1+3]]
                dc[ely,elx] = -penal*(x[ely,elx]**(penal-1))*(Ue.T @ KE @ Ue).item()
        dc = check(nelx, nely, rmin, x, dc)
        x = OC(nelx, nely, x, volfrac, dc)
        change = np.max(np.abs(x - xold))
        if plot:
            plt.imshow(-x,cmap='gray')
            plt.axis('equal')
            plt.axis('off')
            plt.pause(1e-6)
    return x

#%%
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
import random

def upsample_strain_to_nodes(strain_element):
    nely, nelx = strain_element.shape
    strain_node = np.zeros((nely+1, nelx+1))
    # top-left contribution
    strain_node[:-1, :-1] += strain_element
    # top-right
    strain_node[:-1, 1:] += strain_element
    # bottom-left
    strain_node[1:, :-1] += strain_element
    # bottom-right
    strain_node[1:, 1:] += strain_element
    strain_node /= 4
    return strain_node

checkpoint_path = "topopt_dataset_checkpoint.pt"
save_every = 50  # save every 50 samples

# nelx, nely = 80, 40
nelx, nely = 40, 20
volfrac = 0.3
# volfrac = round(random.uniform(0.2, 0.8), 1)
penal = 3
rmin = 1.5
N_samples = 1000

# Preallocate arrays
X_data = np.zeros((N_samples, nely+1, nelx+1, 6), dtype=np.float32)
Y_data = np.zeros((N_samples, nely, nelx), dtype=np.float32)

ndof = 2 * (nelx+1) * (nely+1)

for i in tqdm(range(N_samples)):

    F = sp.lil_matrix((ndof,1))
    auto_force = True

    num_F = random.randint(3,7)
    for _ in range(num_F):
        F[random.randrange(ndof), 0] = random.choice([-1, 1])

    U = FE(nelx, nely, np.ones((nely, nelx))*volfrac, penal, F)
    epsilon_x, epsilon_y, gamma_xy = compute_strains(nelx, nely, U)
    vol_channel = np.ones((nely, nelx)) * volfrac

    x_final = top(nelx, nely, volfrac, penal, rmin, F, plot=False, max_time=90)

    if x_final is None:
        print(f"Sample {i} skipped due to timeout")
        continue

    U_x = U[0:ndof:2]  # take every other starting from 0
    U_y = U[1:ndof:2]  # take every other starting from 1
    U_x_grid = U_x.reshape((nelx+1, nely+1)).T  # transpose to (nely+1, nelx+1)
    U_y_grid = U_y.reshape((nelx+1, nely+1)).T

    epsilon_x_node = upsample_strain_to_nodes(epsilon_x)
    epsilon_y_node = upsample_strain_to_nodes(epsilon_y)
    gamma_xy_node = upsample_strain_to_nodes(gamma_xy)

    volfrac_channel = np.ones((nely+1, nelx+1)) * volfrac

    # Stack into 6-channel input
    X_data[i] = np.stack([U_x_grid, U_y_grid, epsilon_x_node, epsilon_y_node, gamma_xy_node, volfrac_channel], axis=-1)
    
    # Store label
    Y_data[i] = x_final

        # Checkpoint save
    if (i+1) % save_every == 0:
        torch.save({'X': torch.tensor(X_data[:i+1]),
                    'y': torch.tensor(Y_data[:i+1])},
                   checkpoint_path)
        print(f"Checkpoint saved at sample {i+1}")

# Save dataset as a PyTorch file
torch.save({'X': torch.tensor(X_data),
            'y': torch.tensor(Y_data)},
           'topopt_dataset_1000__2025mmdd.pt')