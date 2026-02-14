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
        ### TODO: either make the grid user-defined, or ensure it is sufficient for Roman
        #self.z_mps = np.linspace(0,2,100,endpoint=False)
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
        FPTboost = int(self.accuracyboost - 1.0)
        k_min_dimless = 6.0e-2 # 1e-5 in 1/coverH0, hard coded in cosmolike
        k_max_dimless = 1.0e+6 # 1e+6 in 1/coverH0, hard coded in cosmolike
        N = 350 + 200 * FPTboost
        #self.k = np.logspace(-4, 2, 2000) # in h/Mpc
        self.k_dimless = np.logspace(np.log10(k_min_dimless), np.log10(k_max_dimless), N, endpoint=False)
        self.k = self.k_dimless / self.coverH0 # in h/Mpc
        self.P_window=np.array([.2,.2])  
        self.C_window=.65  
        self.fptmodel = FASTPT(self.k, # treated by FAST-PT as in 1/Mpc, but should be scale-invariant
                               low_extrap=-5, high_extrap=3, 
                               #low_extrap=-7.84, high_extrap=5.65,
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
        ''' Calculate the intrinsic alignment power spectra using FAST-PT at z=0.
        TODO: Do we need other terms in https://fast-pt.readthedocs.io/en/latest/api.html
              e.g. IRres, OV, RSD?
        NOTE: After comparing to cfastpt.c, there are at least some terms missing 
            P_d1d2, P_d2s2, P_d1s2, P_d2s2, P_s2s2 in get_FPT_bias(), that are used 
            in galaxy clustering and galaxy-galaxy lensing.
            Need `one_loop_dd_bias_b3nl` or `one_loop_dd_bias` function below.  
        =============================================================================
        1. Get the linear matter power spectrum from cobaya at the required k and z values
        2. Use FAST-PT to compute the IA power spectra based on the matter power spectrum
        3. Store the IA power spectra in the state dictionary
        4. Return True to indicate successful calculation
        5. The IA power spectra can be accessed later using the get_IA_PS method
        6. The IA power spectra are computed for the following components:
           - IA_tt:  Intrinsic alignment tidal torquing
                includes P_E, P_B
           - IA_ta:  Intrinsic alignment tidal alignment
                includes P_deltaE1, P_deltaE2, P_0E0E, P_0B0B
           - IA_mix: Mixed Intrinsic alignment TA and TT contributions
                includes P_A, P_Btype2, P_DEE, P_DBB
           - one_loop_dd_bias: One-loop galaxy bias corrections
                includes P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4
           - IA_der: Derivative terms k^2 * P(k) of for IA models
                includes P_der
           - IA_ct:  Intrinsic alignment velocity-shear
                includes P_0tE, P_0EtE, P_E2tE, P_tEtE
           - GI_ct:  Galaxy bias x intrinsic alignment velocity-shear
                includes P_d2tE, P_s2tE
           - GI_tt:  Intrinsic alignment 2nd order tidal corrections
                includes P_s2E2, P_d2E2
           - GI_ta:  Intrinsic alignment 2nd order density corrections
                includes P_d2E, P_d20E, P_s2E, P_s20E
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
        # Get the IA power spectra at z=0 and scaling through linear growth factor
        self.mps = self.mpsinterp.P(0, self.k*h0) * (h0**3)  # in (Mpc/h)^3
        # NOTE: FASTPT will use cached results in the input self.mps is unchanged
        #       See https://github.com/jablazek/FAST-PT/blob/master/fastpt/core/FASTPT.py:compute_term
        state['IA_tt']  = self.fptmodel.IA_tt(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['IA_ta']  = self.fptmodel.IA_ta(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['IA_mix'] = self.fptmodel.IA_mix(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['one_loop_dd_bias_b3nl'] = self.fptmodel.one_loop_dd_bias_b3nl(self.mps, 
                                            P_window=self.P_window, C_window=self.C_window)
        ### JX: Below terms are not used in current implementation
        #state['IA_der'] = self.fptmodel.IA_der(self.mps, P_window=self.P_window, C_window=self.C_window)
        #state['IA_ct']  = self.fptmodel.IA_ct(self.mps, P_window=self.P_window, C_window=self.C_window)
        #state['GI_ct']  = self.fptmodel.gI_ct(self.mps, P_window=self.P_window, C_window=self.C_window)
        #state['GI_tt']  = self.fptmodel.gI_tt(self.mps, P_window=self.P_window, C_window=self.C_window)
        #state['GI_ta']  = self.fptmodel.gI_ta(self.mps, P_window=self.P_window, C_window=self.C_window)
        return True

    def get_IA_PS(self):
        ''' Retrieve the computed intrinsic alignment power spectra at z=0.
        =============================================================================
        Returns:
        FPTIA : np.ndarray, (tt_E, tt_B, ta_dE1, ta_dE2, ta_E, ta_B, mixA, mixB, mixEE, mixBB, k, p_lin)
            A 2D array containing the intrinsic alignment power spectra at z=0, with
            each row corresponding to a different component and columns representing 
            k values.
        =============================================================================
        '''
        # NOTE: UNIT CONVERSIONS
        # CAMB: k in 1/Mpc and power spectrum in (Mpc)^3
        # FAST-PT: k in h/Mpc and power spectrum in (Mpc/h)^3 (scale-invariant)
        # CosmoLike: k in c/H0 and power spectrum in (c/H0)^-3
        FPTIA = np.vstack([
                        *self.current_state['IA_tt'], # E, B
                        *self.current_state['IA_ta'], # dE1, dE2, 0E0E, 0B0B
                        *self.current_state['IA_mix'], # A, Btype2, DEE, DBB
                        self.k_dimless, # dimensionless
                        self.mps / (self.coverH0**3), # dimensionless
                        # self.k,
                        # self.current_state['IA_der'],
                        # *self.current_state['IA_tt'],
                        # *self.current_state['IA_mix'],
                        # *self.current_state['IA_ta'],
                        # *self.current_state['IA_ct'],
                        # *self.current_state['GI_ct'],
                        # *self.current_state['GI_ta'],
                        # *self.current_state['GI_tt'], #check if repeated
                        ])
        FPTIA[:-2,:] /= (self.coverH0**3) # convert from (Mpc/h)^3 to dimensionless
        return FPTIA
    
    def get_bias_PS(self):
        ''' Retrieve the computed galaxy-intrinsic alignment power spectra at z=0.
        =============================================================================
        Returns:
        FPTbias : np.ndarray, (d1d2, d2d2, d1s2, d2s2, s2s2, d1p3, k, p_lin)
            A 2D array containing the intrinsic alignment power spectra at z=0, with
            each row corresponding to a different component and columns representing 
            k values.
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
        #print(f'sigma4 term is {self.current_state["one_loop_dd_bias"][7]}!!!')
        FPTbias[:-2,:] /= (self.coverH0**3) # convert from (Mpc/h)^3 to dimensionless
        return FPTbias, self.current_state["one_loop_dd_bias_b3nl"][7]/(self.coverH0**3)