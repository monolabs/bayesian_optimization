import numpy as np

import torch
tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          }
from torch.quasirandom import SobolEngine

from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.acquisition.objective import MCAcquisitionObjective

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood



def smooth_objective(y: torch.Tensor, lower:float=None, upper:float=None, steepness:float=1) -> torch.Tensor:
    assert lower is not None or upper is not None, 'lower and upper cannot be both None'
    
    if lower is not None:
        sig_1 = 1/(1+torch.exp(-steepness * (y - lower)))
        if upper is not None:
            assert lower < upper, 'upper must be higher than lower'
            sig_2 = 1 - 1/(1+torch.exp(-steepness * (y - upper)))
            return sig_1*sig_2
        return sig_1
    else:
        sig_2 = 1 - 1/(1+torch.exp(-steepness * (y - upper)))
        return sig_2
    
    
def sharp_objective(y: torch.Tensor, target_bounds:list, bounds:list) -> torch.Tensor:
    assert target_bounds[0] > bounds[0] and bounds[1] > target_bounds[1], 'target_bounds must be inside bounds'
    
    l = target_bounds[1] - target_bounds[0]
    c = (target_bounds[1] + target_bounds[0])/2
    
    grad_1 = 1/(target_bounds[0] - bounds[0])
    grad_2 = 1/(bounds[1] - target_bounds[1])
    grad = min([grad_1, grad_2])
    
    obj = 1-torch.abs(grad*(y-c)) + l*grad/2
    obj[obj>=1] = 1
    obj[obj<0] = 0
    return obj


def hybrid_objective(y:torch.Tensor, lower:float, upper:float, steepness:float=1.0)->torch.Tensor:
    assert lower is not None or upper is not None, 'lower and upper cannot be both None'
    obj = torch.ones(y.shape).to(y)
    
    if lower is not None:
        # < lower
        mask = y < lower
        obj[mask] = torch.exp(-steepness*(-y[mask]+lower))
        if upper is not None:
            assert lower < upper, 'upper must be higher than lower'
            # > upper
            mask = y > upper
            obj[mask] = torch.exp(-steepness*(y[mask]-upper))
            return obj
        return obj
    else:
        # > upper
        mask = y > upper
        obj[mask] = torch.exp(-steepness*(y[mask]-upper))
        return obj