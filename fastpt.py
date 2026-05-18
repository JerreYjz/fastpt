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
        ''' 
          Initialize the FASTPT cobaya theory module
        '''
        super().initialize()
                
        # ----------------------------------------------------------------------
        # k conventions:
        # FAST-PT: The input k-grid in 1/Mpc. Must be logarithmically spaced 
        #          with equal spacing and contain an even number of elements.
        # Cobaya:  (k, P(k)) in  units of 1/Mpc, Mpc^3 respectively
        # ----------------------------------------------------------------------
        
        self.accuracyboost  = float(self.extra_args.get("accuracyboost", 1.0))
        
        self.kmax_boltzmann = self.extra_args.get("kmax_boltzmann", 7.5) # 1/Mpc
        
        self.extrap_kmax   = self.extra_args.get("extrap_kmax", 250.0) # 1/Mpc
        
        FPTboost = np.max(int(self.accuracyboost - 1.0), 0)
                
        self.k_cutoff = 1.0e4 / 2997.92458  # units: h/Mpc
                
        self.k = np.logspace(np.log10(0.05), 
                             np.log10(1.0e+6), 
                             1100 + 200 * FPTboost, 
                             endpoint=False) # dimensionless (cosmolike units)
        self.k = self.k / 2997.92458  # units: in h/Mpc
        
        self.P_window = np.array([.2, .2]) 
        
        self.C_window = .65  
        
        self.fptmodel = FASTPT(self.k, 
                               low_extrap = -5, 
                               high_extrap = 3, 
                               n_pad = int(0.5*len(self.k)))

    def get_requirements(self):
      return {  
        "H0": None,
        "Pk_interpolator": {
          "z": np.array([0.0, ]),
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": False,
          "vars_pairs": [("delta_tot", "delta_tot")]
        }
      }

    def calculate(self, state, want_derived=False, **par):
      ''' 
      Calculate perturbation terms at z=0 for intrinsic alignment and galaxy.    
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
      Parameters:
          state : dict
              A dictionary to store the computed IA power spectra at z=0.
          want_derived : bool, optional
              A flag indicating whether derived quantities are requested (default is False).
          **par : dict
              Additional parameters (not used in this implementation).
      
      Returns:
          bool
              True if the calculation was successful, False otherwise.
      '''
      mps = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                              nonlinear = False, 
                                              extrap_kmin = 1e-6,
                                              extrap_kmax = self.extrap_kmax*self.accuracyboost)
      h0 = par["H0"]/100.0
      
      self.mps = mps.P(0, self.k*h0) * (h0**3)  # in (Mpc/h)^3
      
      state['IA_tt']  = self.fptmodel.IA_tt(self.mps, 
                                            P_window = self.P_window, 
                                            C_window = self.C_window)
      state['IA_ta']  = self.fptmodel.IA_ta(self.mps, 
                                            P_window = self.P_window, 
                                            C_window = self.C_window)
      state['IA_mix'] = self.fptmodel.IA_mix(self.mps,
                                             P_window = self.P_window, 
                                             C_window = self.C_window)
      state['one_loop_dd_bias_b3nl'] = self.fptmodel.one_loop_dd_bias_b3nl(self.mps, 
                                                                           P_window = self.P_window, 
                                                                           C_window = self.C_window)
      return True

    def get_IA_PS(self):
      ''' 
      Retrieve the intrinsic alignment power spectra at z=0.
      
      Returns:
        FPTIA : np.ndarray of shape (12, Nk) 
          (tt_E, tt_B, ta_dE1, ta_dE2, ta_E, ta_B, mixA, mixB, mixEE, mixBB, k, Plin)
          A 2D array containing the intrinsic alignment power spectra at z=0, with
          each row corresponding to a different component and columns representing k values.
        k_cutoff : float
          The cutoff scale in h/Mpc beyond which the PS spectra may not be reliable.
      '''
      FPTIA = np.vstack([*self.current_state['IA_tt'],    # tt_E, tt_B
                         *self.current_state['IA_ta'],    # ta_dE1, ta_dE2, ta_0E0E, ta_0B0B
                         *self.current_state['IA_mix'],   # mixA, mixBtype2, mixDEE, mixDBB
                         self.k,                          # dimensionless
                         self.mps,                        # dimensionless
                        ])
      return FPTIA, self.k_cutoff
    
    def get_bias_PS(self):
      ''' 
        Retrieve the galaxy 1-loop power spectra at z=0.
      
        Returns:
          FPTbias : np.ndarray of shape (8, Nk)
              (d1d2, d2d2, d1s2, d2s2, s2s2, d1p3, k, p_lin)
              A 2D array containing the intrinsic alignment power spectra at z=0, with
              each row corresponding to a different component and columns representing 
              k values.
          sigma4: float
              Nomalization factor for various perturbation terms (low-k limit)
      '''
      FPTbias = np.vstack([self.current_state['one_loop_dd_bias_b3nl'][2], # d1d2
                           self.current_state['one_loop_dd_bias_b3nl'][3], # d2d2
                           self.current_state['one_loop_dd_bias_b3nl'][4], # d1s2
                           self.current_state['one_loop_dd_bias_b3nl'][5], # d2s2
                           self.current_state['one_loop_dd_bias_b3nl'][6], # s2s2
                           self.current_state['one_loop_dd_bias_b3nl'][8], # d1p3
                           self.k,
                           self.mps,
                          ])
      return FPTbias, self.current_state["one_loop_dd_bias_b3nl"][7]
