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
        self.fptmodel = FASTPT(self.k, to_do=['IA_tt'], low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(self.k)))
        self.handler = FPTHandler(self.fptmodel)
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
        
        
        self.handler.update_default_params(P=self.mps, P_window=np.array([0.2, 0.2]), C_window=0.75)
        state['IA_tt'] = self.handler.run("IA_tt")
        #state['IA_ta'] = self.handler.run("IA_ta")
        #state['IA_der'] = self.handler.run("IA_der")
        #state['IA_ct'] = self.handler.run("IA_ct")
        #state['IA_mix'] = self.handler.run("IA_mix")

        return True

    def get_IA_PS(self):
        IA = np.zeros((3,len(self.k)))
        IA[0] = self.k
        IA[1] = self.current_state['IA_tt'][0]
        IA[2] = self.current_state['IA_tt'][1]
        return IA


    	