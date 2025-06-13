import numpy as np
from math import factorial

class Waveguide:
    def __init__(self, material, betas: np.ndarray, etas: np.ndarray):
        self.material = material
        self.betas = betas
        self.etas = etas
        self.num_modes = betas.shape[1]
        if material == "SiN":
            self.n2 = 2.4e-19
        elif material == "Si":
            self.n2 = 5.6e-18
        else:
            raise ValueError("Material not supported. Choose 'SiN' or 'Si'.")

    def get_dispersion(self, omegas: np.ndarray):

        # The dispersion term in frequency space
        DFR = np.zeros((self.num_modes, omegas.size), dtype=complex)
        for mode_i in range(self.num_modes):
            DFR[mode_i, :] = 1j * (self.betas[0, mode_i] - np.real(self.betas[0, 0])) + \
                            1j * (self.betas[1, mode_i] - np.real(self.betas[1, 0])) * omegas
        
        for mode_i in range(self.num_modes):
            for order in range(2, self.betas.shape[0]):
                DFR[mode_i, :] += 1j * (self.betas[order, mode_i] / factorial(order)) * (omegas ** order)
        return DFR

    def get_etas(self):
        count_plmn = 0
        count_mn = 0
        nonzero_idx_plmn_list = []
        nonzero_idx_mn_list = []
        SR_list = []
        for p in range(self.num_modes):
            for l in range(self.num_modes):
                for m in range(self.num_modes):
                    for n in range(self.num_modes):
                        if self.etas[p, l, m, n] == 0:
                            continue
                        nonzero_idx_plmn_list.append([p, l, m, n])

                        SR_list.append(self.etas[p, l, m, n])
                        count_plmn += 1
                
                        # Only avoid a set of two indices if all the other possible
                        # combinations have 0 values in SR
                        if np.all(self.etas[:, :, m, n] == 0):
                            continue
                        nonzero_idx_mn_list.append([p, l])
                        count_mn += 1

        self.nonzero_idx_plmn = np.array(nonzero_idx_plmn_list, dtype=np.uint32).T
        self.nonzero_idx_mn = np.array(nonzero_idx_mn_list, dtype=np.uint32).T
        self.SR = np.array(SR_list, dtype=np.complex128).T

        return self.nonzero_idx_plmn, self.nonzero_idx_mn, self.SR
