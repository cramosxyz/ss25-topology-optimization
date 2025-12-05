import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

class SIMPOptimizer:
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        """
        Initialize the SIMP Topology Optimizer.
        """
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        
        # Initialize density distribution
        self.x = np.ones((nely, nelx)) * volfrac
        
        # Pre-compute element stiffness matrix (constant)
        self.KE = self._compute_element_stiffness()
        
        # Total Degrees of Freedom
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)

    def _compute_element_stiffness(self):
        """gets K for each QUAD4 element"""
        E = 1.0
        nu = 0.3
        k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                      -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
        
        KE = E / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        return KE

    def _solve_fem(self, F):
        """Previously FE(). Solves KU=F."""
        K = sp.lil_matrix((self.ndof, self.ndof))
        U = np.zeros((self.ndof, 1))
        
        # Assemble K
        # Note: Vectorizing this loop is a common optimization, 
        # but kept as original for clarity/consistency with your logic.
        for elx in range(self.nelx):
            for ely in range(self.nely):
                n1 = (self.nely+1)*elx + ely
                n2 = (self.nely+1)*(elx+1) + ely
                edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
                K[np.ix_(edof, edof)] += self.x[ely, elx]**self.penal * self.KE
                
        K = K.tocsr()
        
        # Fixities: left edge
        fixeddofs = np.arange(0, 2*(self.nely+1), 1)
        alldofs = np.arange(self.ndof)
        freedofs = np.setdiff1d(alldofs, fixeddofs)
        
        # Solve
        # Flatten F to ensure dimensions match for spsolve
        F_vec = F.toarray().flatten() if sp.issparse(F) else F.flatten()
        
        U[freedofs, 0] = spla.spsolve(K[freedofs, :][:, freedofs], F_vec[freedofs])
        U[fixeddofs, 0] = 0
        return U

    def _filter_sensitivity(self, dc):
        """Previously check(). Mesh-independency filter."""
        dcn = np.zeros((self.nely, self.nelx))
        for i in range(self.nelx):
            for j in range(self.nely):
                s = 0
                # Optimized range limits
                i_min = max(i - int(self.rmin), 0)
                i_max = min(i + int(self.rmin) + 1, self.nelx)
                j_min = max(j - int(self.rmin), 0)
                j_max = min(j + int(self.rmin) + 1, self.nely)
                
                for k in range(i_min, i_max):
                    for l in range(j_min, j_max):
                        fac = self.rmin - np.sqrt((i-k)**2 + (j-l)**2)
                        s += max(0, fac)
                        dcn[j, i] += max(0, fac) * self.x[l, k] * dc[l, k]
                dcn[j, i] /= (self.x[j, i] * s)
        return dcn

    def _update_distribution(self, dc):
        """Previously OC(). Optimality Criteria update."""
        l1, l2 = 0, 1e5
        move = 0.2
        xnew = np.zeros_like(self.x)
        
        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            # Using numpy broadcasting for speed
            term = self.x * np.sqrt(-dc / lmid)
            xnew = np.maximum(0.001, np.maximum(self.x - move, np.minimum(1.0, np.minimum(self.x + move, term))))
            
            if np.sum(xnew) - self.volfrac * self.nelx * self.nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        return xnew

    def compute_strains(self, U):
        """Compute element-wise strains from nodal displacements."""
        U_x = U[0::2, 0]
        U_y = U[1::2, 0]

        epsilon_x = np.zeros((self.nely, self.nelx))
        epsilon_y = np.zeros((self.nely, self.nelx))
        gamma_xy = np.zeros((self.nely, self.nelx))

        for elx in range(self.nelx):
            for ely in range(self.nely):
                n1 = (self.nely+1)*elx + ely
                n2 = (self.nely+1)*(elx+1) + ely
                n3 = (self.nely+1)*(elx+1) + (ely+1)
                n4 = (self.nely+1)*elx + (ely+1)

                u_e = np.array([U_x[n1], U_y[n1],
                                U_x[n2], U_y[n2],
                                U_x[n3], U_y[n3],
                                U_x[n4], U_y[n4]])

                epsilon_x[ely, elx] = 0.5*((u_e[2] - u_e[0]) + (u_e[4] - u_e[6]))
                epsilon_y[ely, elx] = 0.5*((u_e[5] - u_e[1]) + (u_e[7] - u_e[3]))
                gamma_xy[ely, elx] = 0.5*((u_e[3] - u_e[1]) + (u_e[2] - u_e[0]))

        return epsilon_x, epsilon_y, gamma_xy

    @staticmethod
    def upsample_strain_to_nodes(strain_element):
        nely, nelx = strain_element.shape
        strain_node = np.zeros((nely+1, nelx+1))
        strain_node[:-1, :-1] += strain_element
        strain_node[:-1, 1:] += strain_element
        strain_node[1:, :-1] += strain_element
        strain_node[1:, 1:] += strain_element
        strain_node /= 4
        return strain_node

    def optimize(self, F, max_time=60, plot=False, min_change=0.01):
        """
        Main optimization loop.
        """
        start_time = time.time()
        change = 1.0
        loop = 0
        
        if plot:
            plt.ion() # Interactive mode on
            fig, ax = plt.subplots()
            im = ax.imshow(-self.x, cmap='gray', interpolation='none')
            ax.axis('off')
        
        print(f"Starting optimization with {self.nelx}x{self.nely} elements...")

        while change > min_change:
            loop += 1
            # Timer check
            if time.time() - start_time > max_time:
                print("\nSIMP aborted: exceeded max_time")
                break

            xold = self.x.copy()
            
            # 1. Finite Element Analysis
            U = self._solve_fem(F)
            
            # 2. Sensitivity Analysis
            dc = np.zeros((self.nely, self.nelx))
            
            # Vectorizing the sensitivity calculation is harder without major refactoring,
            # so we keep the loop but optimized access slightly.
            for ely in range(self.nely):
                for elx in range(self.nelx):
                    n1 = (self.nely+1)*elx + ely
                    n2 = (self.nely+1)*(elx+1) + ely
                    # Map indices
                    idx = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3]
                    Ue = U[idx]
                    # Compliance sensitivity
                    dc[ely, elx] = -self.penal * (self.x[ely, elx]**(self.penal-1)) * (Ue.T @ self.KE @ Ue).item()
            
            # 3. Filter Sensitivity
            dc = self._filter_sensitivity(dc)
            
            # 4. Update Densities (Optimizer)
            self.x = self._update_distribution(dc)
            
            # Convergence check
            change = np.max(np.abs(self.x - xold))
            
            # Print status every 5 iterations or if change is large
            if loop % 5 == 0 or loop == 1:
                print(f" It.: {loop:4d} | Obj.: {np.sum(self.x):.3f} | Vol.: {np.mean(self.x):.3f} | ch.: {change:.3f}")

            # Plotting
            if plot:
                im.set_array(-self.x)
                plt.draw()
                plt.pause(0.01)

        print("Optimization finished.")
        if plot:
            plt.show(block=True)
            
        return self.x, U