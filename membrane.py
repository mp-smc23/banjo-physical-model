import numpy as np

class Membrane:
    def __init__(self, k, sigma_0=1.4, sigma_1=0.005,
                 thickness=0.000355, poissions_ratio=0.4,
                 tension=4000, young_modulus=3800000000,
                 material_density=1380):
        self.k = k
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.H = thickness
        self.nu = poissions_ratio
        self.T = tension
        self.E = young_modulus
        self.rho = material_density
        self.calculate_physical_coefs()
        self.calculate_coefs()
        self.print_parameters()
        self.check_stability_conditions()

    def calculate_physical_coefs(self):
        self.D = (self.E * (self.H**3)) / (12 * (1 - self.nu**2))

        self.c = np.sqrt(self.T / (self.rho * self.H))
        self.h = self.c * self.k
        self.L = 0.25
        self.M = 11
        self.h = self.L / self.M

        self.kappa = np.sqrt(self.D / (self.rho * self.H))

    def calculate_coefs(self):
        self.alpha = ((self.c * self.k) / self.h)**2
        self.beta = (self.k * self.kappa)**2 / self.h**4
        self.gamma = 2 * self.sigma_0 * self.k
        self.delta = (2 * self.sigma_1 * self.k) / self.h**2

    def check_stability_conditions(self):
        ck_sq = (self.c*self.k)**2
        sq_k_4 = 4 * self.sigma_1*self.k
        stability_s = np.sqrt((ck_sq + sq_k_4 + np.sqrt((ck_sq + sq_k_4)**2 + 16*(self.kappa*self.k)**2)))

        print(f"h = {self.h} >= {stability_s} = stability_s")
        if self.h >= stability_s:
            print("Stability condition is satisfied")
        else:
            raise ValueError("Stability condition is NOT satisfied!!!")
        print("=====================================")

    def print_parameters(self):
        print("Banjo Membrane Parameters")
        print(f"k = {self.k}, sigma_0 = {self.sigma_0}, sigma_1 = {self.sigma_1}, thickness = {self.H}, poissions_ratio = {self.nu}, tension = {self.T}, young_modulus = {self.E}, material_density = {self.rho}")
        print(f"D = {self.D}, c = {self.c}, h = {self.h}, L = {self.L}, M = {self.M}, kappa = {self.kappa}")
        print(f"alpha = {self.alpha}, beta = {self.beta}, gamma = {self.gamma}, delta = {self.delta}")
        print("=====================================")

    @staticmethod
    def wave(w, l, m, n=1):
        return w[l+1,m,n] + w[l-1,m,n] + w[l,m+1,n] + w[l,m-1,n] - 4*w[l,m,n] 

    @staticmethod
    def stiff(w, l, m, n=1):
        return 20*w[l,m,n] - 8*(w[l+1,m,n] +w[l-1,m,n] + w[l,m+1,n] + w[l,m-1,n]) + 2*(w[l+1,m+1,n] + w[l-1,m+1,n] + w[l+1,m-1,n] + w[l-1,m-1,n]) + (w[l+2,m,n] + w[l-2,m,n] + w[l,m+2,n] + w[l,m-2,n]) 

    @staticmethod
    def damp(w, l, m, n=1):
        return w[l,m,n] - w[l,m,n+1]

    @staticmethod
    def freq_damp(w, l, m, n=1):
        return Membrane.wave(w,l,m,n) - Membrane.wave(w,l,m,n+1)

    def fd(self, w):
        for l in range(2,self.M-2):
            for m in range(2,self.M-2):
                w[l, m, 0] = (
                    2 * w[l, m, 1]
                    - w[l, m, 2]
                    + self.alpha * Membrane.wave(w, l, m)
                    - self.beta * Membrane.stiff(w, l, m)
                    - self.gamma * Membrane.damp(w, l, m)
                    + self.delta * Membrane.freq_damp(w, l, m)
                )
        
        return w
