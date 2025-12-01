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
        super().initialize()

        self.z_mps = np.linspace(0,2,100,endpoint=False)
        self.k = np.logspace(-4, 2, 2000)
        self.P_window=np.array([.2,.2])  
        self.C_window=.65  
        self.fptmodel = FASTPT(self.k,
                               low_extrap=-5, high_extrap=3, 
                               n_pad=int(0.5*len(self.k)))
        
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
              "k_max": 250,
              "nonlinear": (True,False),
              "vars_pairs": ([("delta_tot", "delta_tot")])
          },
          "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
          }
        } 





    def get_requirements(self):
        return self.req

    def calculate(self, state, want_derived=False, **par):
        #####Check if MPS is provided#####
        self.mpsinterp = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmax=2.5e2)
        self.mps = self.mpsinterp.P(0, self.k)
        
        
        state['IA_tt'] = self.fptmodel.IA_tt(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['IA_ta'] = self.fptmodel.IA_ta(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['IA_der'] = self.fptmodel.IA_der(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['IA_ct'] = self.fptmodel.IA_ct(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['IA_mix'] = self.fptmodel.IA_mix(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['GI_ct'] = self.fptmodel.gI_ct(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['GI_tt'] = self.fptmodel.gI_tt(self.mps, P_window=self.P_window, C_window=self.C_window)
        state['GI_ta'] = self.fptmodel.gI_ta(self.mps, P_window=self.P_window, C_window=self.C_window)



        return True

    def get_IA_PS(self):
        IA = np.vstack([
                        self.k,
                        self.current_state['IA_der'],
                        *self.current_state['IA_tt'],
                        *self.current_state['IA_mix'],
                        *self.current_state['IA_ta'],
                        *self.current_state['IA_ct'],
                        *self.current_state['GI_ct'],
                        *self.current_state['GI_ta'],
                        *self.current_state['GI_tt'], #check if repeated
                        ])


        return IA


    	