import fastpt as fpt
from fastpt import FASTPT, FPTHandler
import os.path as path
import numpy as np
from cobaya.theory import Theory
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d

class fastpt(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str
    
    def initialize(self):
        ''' Initialize the FASTPT cobaya theory module
        '''
        super().initialize()
        self.coverH0 = 2997.92458 # Mpc/h
        ### Set up matter power spectrum interpolation grids
        self.z_mps = np.array([0.0, ]) # only need z=0 for IA power spectra

        #### wavenumbers conventions:
        # FAST-PT: The input k-grid in 1/Mpc (unit-invariant actually...). 
        #          Must be logarithmically spaced with equal spacing in log(k) and 
        #          contain an even number of elements.
        # Cobaya:  In the interpolator returned, the input k and resulting P(k,z) are in 
        #          units of 1/Mpc and Mpc^3 respectively (not h^-1 in units).
        # CosmoLike: k in c/H0 and power spectrum in (c/H0)^-3, in the FPT structure
        self.accuracyboost = float(self.extra_args.get("accuracyboost", 1.0))
        self.kmax_boltzmann = self.extra_args.get("kmax_boltzmann", 7.5) # 1/Mpc
        self.extrap_kmax = self.extra_args.get("extrap_kmax", 250.0) # 1/Mpc
        FPTboost = np.max(int(self.accuracyboost - 1.0), 0)
        ### The kmin, kmax, and number of grids are tuned for Roman accuracy 
        ### May need to revisit as survey configuration and analysis choices evolve.
        k_min_dimless = 1.0e-1
        k_max_dimless = 1.0e+6
        self.k_cutoff_dimless = 1.0e4 
        N = 1000 + 200 * FPTboost
        
        self.k_dimless = np.logspace(np.log10(k_min_dimless), np.log10(k_max_dimless), N, 
                                     endpoint=False)
        self.k = self.k_dimless / self.coverH0 # in h/Mpc
        self.P_window=np.array([.2,.2]) 
        self.C_window=.65  
        self.fptmodel = FASTPT(self.k,
                               low_extrap=-5, high_extrap=3, 
                               n_pad=int(0.5*len(self.k)))
        
        ### Set up requirements for cobaya
        ### Need linear matter power spectrum and cosmological params from CAMB
        self.req = {
          "H0": None,
          "omegabh2": None,
          "omegach2": None,
          "As": None,
          "ns": None,
          "mnu": None,
          "w": None,
          "Pk_interpolator": {
              "z": self.z_mps,
              "k_max": self.kmax_boltzmann * self.accuracyboost,
              "nonlinear": False,
              "vars_pairs": [("delta_tot", "delta_tot")]
          },
          "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
          }
        }

    def get_requirements(self):
        return self.req

    def calculate(self, state, want_derived=False, **par):
        ''' Calculate perturbation terms at z=0 for intrinsic alignment and galaxy.
        TODO: Double check if we are returning all the terms needed at 1-loop level.
        =============================================================================
        1. Get the linear matter power spectrum from cobaya at the required k and z values
        2. Use FAST-PT to compute the IA and galaxy power spectra at higher orders
        3. Store the IA power spectra in the state dictionary
        4. Return True to indicate successful calculation
        5. The IA/galaxy power spectra can be accessed with get_IA/bias_PS
        6. The IA/galaxy power spectra are computed for the following components:
           - IA_ta:  Intrinsic alignment from tidal alignment (GI+II)
                includes P_deltaE1, P_deltaE2, P_0E0E, P_0B0B
           - IA_tt:  Intrinsic alignment from tidal torquing (GI+II)
                includes P_E, P_B
           - IA_mix: Intrinsic alignment from TA x TT (II)
                includes P_A, P_Btype2, P_DEE, P_DBB
           - one_loop_dd_bias_b3nl: One-loop galaxy bias corrections, including b3nl
                includes P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, Pd1p3
        =============================================================================
        Parameters:
        state : dict
            A dictionary to store the computed IA power spectra at z=0.
        want_derived : bool, optional
            A flag indicating whether derived quantities are requested (default is False).
        **par : dict
            Additional parameters (not used in this implementation).
        =============================================================================
        Returns:
        bool
            True if the calculation was successful, False otherwise.
        =============================================================================
        '''
        #####Check if MPS is provided#####
        self.mpsinterp = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                            nonlinear=False, 
                            extrap_kmax=self.extrap_kmax * self.accuracyboost)
        h0 = par["H0"]/100.0
        # Get the IA & galaxy power spectra at z=0 and scaling through linear growth factor
        self.mps = self.mpsinterp.P(0, self.k*h0) * (h0**3)  # in (Mpc/h)^3
        # NOTE: FASTPT will use cached results if the input self.mps is unchanged
        # See https://github.com/jablazek/FAST-PT/blob/master/fastpt/core/FASTPT.py
        state['IA_tt']  = self.fptmodel.IA_tt(self.mps, 
                            P_window=self.P_window, C_window=self.C_window)
        state['IA_ta']  = self.fptmodel.IA_ta(self.mps, 
                            P_window=self.P_window, C_window=self.C_window)
        state['IA_mix'] = self.fptmodel.IA_mix(self.mps, 
                            P_window=self.P_window, C_window=self.C_window)
        state['one_loop_dd_bias_b3nl'] = self.fptmodel.one_loop_dd_bias_b3nl(self.mps, 
                            P_window=self.P_window, C_window=self.C_window)
        return True

    def get_IA_PS(self):
        ''' Retrieve the intrinsic alignment power spectra at z=0.
        =============================================================================
        Returns:
        FPTIA : np.ndarray of shape (12, Nk) 
            (tt_E, tt_B, ta_dE1, ta_dE2, ta_E, ta_B, mixA, mixB, mixEE, mixBB, k, Plin)
            A 2D array containing the intrinsic alignment power spectra at z=0, with
            each row corresponding to a different component and columns representing 
            k values.
        k_cutoff_dimless : float
            The cutoff scale in dimensionless units (k * c/H0) beyond which the power
            spectra may not be reliable.
        =============================================================================
        '''
        # NOTE: UNIT CONVERSIONS
        # CAMB: k in 1/Mpc and power spectrum in (Mpc)^3
        # FAST-PT: k in h/Mpc and power spectrum in (Mpc/h)^3 (scale-invariant)
        # CosmoLike: k in c/H0 and power spectrum in (c/H0)^-3
        FPTIA = np.vstack([
                        *self.current_state['IA_tt'], # tt_E, tt_B
                        *self.current_state['IA_ta'], # ta_dE1, ta_dE2, ta_0E0E, ta_0B0B
                        *self.current_state['IA_mix'], # mixA, mixBtype2, mixDEE, mixDBB
                        self.k_dimless, # dimensionless
                        self.mps / (self.coverH0**3), # dimensionless
                        ])
        FPTIA[:-2,:] /= (self.coverH0**3) # convert from (Mpc/h)^3 to dimensionless
        # Apply Gaussian cutoff 
        # FPTIA[:-2,:] *= np.exp(-(self.k_dimless / self.k_cutoff_dimless) ** 2) 
        return FPTIA, self.k_cutoff_dimless
    
    def get_bias_PS(self):
        ''' Retrieve the galaxy 1-loop power spectra at z=0.
        =============================================================================
        Returns:
        FPTbias : np.ndarray of shape (8, Nk)
            (d1d2, d2d2, d1s2, d2s2, s2s2, d1p3, k, p_lin)
            A 2D array containing the intrinsic alignment power spectra at z=0, with
            each row corresponding to a different component and columns representing 
            k values.
        sigma4: float
            Nomalization factor for various perturbation terms (low-k limit)
        =============================================================================
        '''
        # NOTE: UNIT CONVERSIONS
        # CAMB: k in 1/Mpc and power spectrum in (Mpc)^3
        # FAST-PT: k in h/Mpc and power spectrum in (Mpc/h)^3 (scale-invariant)
        # CosmoLike: k in c/H0 and power spectrum in (c/H0)^-3
        FPTbias = np.vstack([
                        self.current_state['one_loop_dd_bias_b3nl'][2], # d1d2
                        self.current_state['one_loop_dd_bias_b3nl'][3], # d2d2
                        self.current_state['one_loop_dd_bias_b3nl'][4], # d1s2
                        self.current_state['one_loop_dd_bias_b3nl'][5], # d2s2
                        self.current_state['one_loop_dd_bias_b3nl'][6], # s2s2
                        self.current_state['one_loop_dd_bias_b3nl'][8], # d1p3
                        self.k_dimless, # dimensionless
                        self.mps / (self.coverH0**3), # dimensionless
                        ])
        FPTbias[:-2,:] /= (self.coverH0**3) # convert from (Mpc/h)^3 to dimensionless
        return FPTbias, self.current_state["one_loop_dd_bias_b3nl"][7]/(self.coverH0**3)