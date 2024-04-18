import numpy as np

class BanjoString:
    def __init__(self, k, frequency, 
                 sigma_0=1, sigma_1=0.009, 
                 radius=0.0025, L=0.67, 
                 young_modulus=210000000000, material_density=7962):
        # A string in a guitar is made of steel (density 7962kg/m3). 
        # It is 63.5 cm long, and has diameter of 0.4 mm.
        self.k = k
        self.f0 = frequency
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.r = radius
        self.L = L
        self.E = young_modulus # 210 GPa
        self.rho = material_density
        
        self.calculate_physical_coefs()
        self.calculate_coefs()
        self.print_parameters()
        self.check_stability_conditions()
        

    def calculate_physical_coefs(self):
        self.I = np.pi * self.r**4 / 4
        self.A = np.pi * self.r**2

        self.T = 4 * self.f0**2 * self.L**2 * np.pi * (0.5*self.r)**2 * self.rho
        self.c = np.sqrt(self.T/(self.rho*self.A))        
        self.h = self.c * self.k
        self.N = int(np.floor(self.L / self.h))
        self.h = self.L / self.N

        self.kappa = np.sqrt((self.E * self.I) / (self.rho * self.A))

    def calculate_coefs(self):
        self.alpha = ((self.c * self.k) / self.h)**2
        self.beta = self.kappa**2 * self.k**2 / self.h**4
        self.gamma = 2 * self.sigma_0 * self.k
        self.delta = (2 * self.sigma_1 * self.k) / self.h**2

    def check_stability_conditions(self):
        ck_sq = (self.c*self.k)**2
        sq_k_4 = 4 * self.sigma_1*self.k
        stability_s = np.sqrt(0.5 * (ck_sq + sq_k_4 + np.sqrt((ck_sq + sq_k_4)**2 + 16*(self.kappa*self.k)**2)))

        print(f"h = {self.h} >= {stability_s} = stability_s")
        if self.h >= stability_s:
            print("Stability condition is satisfied")
        else:
            raise ValueError("Stability condition is NOT satisfied!!!" + f" h = {self.h} < {stability_s} = stability_s)")
        print("=====================================")

    def print_parameters(self):
        print("Banjo String Parameters")
        print(f"k = {self.k}, frequency = {self.f0}, sigma_0 = {self.sigma_0}, sigma_1 = {self.sigma_1}, radius = {self.r}, L = {self.L}, young_modulus = {self.E}, material_density = {self.rho}")
        print(f"I = {self.I}, A = {self.A}, c = {self.c}, tension = {self.T}, h = {self.h}, N = {self.N}, kappa = {self.kappa}")
        print(f"alpha = {self.alpha}, beta = {self.beta}, gamma = {self.gamma}, delta = {self.delta}")
        print("=====================================")

    @staticmethod
    def wave(u, l, n=1):
        return u[l+1,n] - 2*u[l,n] + u[l-1,n]

    @staticmethod
    def stiff(u,l,n=1):
        return u[l+2,n] - 4*u[l+1,n] + 6*u[l,n] - 4*u[l-1,n] + u[l-2,n]

    @staticmethod
    def damp(u, l,n=1):
        return u[l,n] - u[l,n+1]

    @staticmethod
    def freq_damp(u, l, n=1):
        return BanjoString.wave(u,l,n) - BanjoString.wave(u,l,n+1)

    def fd(self, u):
        for l in range(2, self.N - 2):
            u[l, 0] = (
                2 * u[l, 1]
                - u[l, 2]
                + self.alpha * BanjoString.wave(u, l)
                - self.beta * BanjoString.stiff(u, l)
                - self.gamma * BanjoString.damp(u, l)
                + self.delta * BanjoString.freq_damp(u, l)
            )

        # Boundry Conditions, first and last element are 0, -1 and N+1 are same as second and second to last but opposite sign
        u[1, 0] = (
            2 * u[1, 1]
            - u[1, 2]
            + self.alpha * BanjoString.wave(u, 1)
            - self.beta * (u[3, 1] - 4 * u[2, 1] + 5 * u[1, 1])
            - self.gamma * BanjoString.damp(u, 1)
            + self.delta * BanjoString.freq_damp(u, 1)
        )
        u[self.N - 2, 0] = (
            2 * u[self.N - 2, 1]
            - u[self.N - 2, 2]
            + self.alpha * BanjoString.wave(u, self.N - 2)
            - self.beta * (5 * u[self.N - 2, 1] - 4 * u[self.N - 3, 1] + u[self.N - 4, 1])
            - self.gamma * BanjoString.damp(u, self.N - 2)
            + self.delta * BanjoString.freq_damp(u, self.N - 2)
        )

        return u
