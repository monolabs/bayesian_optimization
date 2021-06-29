from contextlib import ExitStack

import numpy as np

import torch
tkwargs = {"dtype": torch.double,
           "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#            "device": torch.device("cpu")
          }
from torch.quasirandom import SobolEngine

from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.optim.optimize import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.generation.sampling import MaxPosteriorSampling

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
import gpytorch.settings as gpts

from ModifiedSingleTaskGP import ModifiedSingleTaskGP


def modified_neg_ackley(X: torch.Tensor, C: int) -> torch.Tensor:
    '''
    compute modified ackley function as in:
    "Bayesian Optimization for Categorical and Category-Specific Continuous Inputs"
    - Dang Nguyen, Sunil Gupta, Santu Rana, Alistair Shilton, Svetha Venkatesh -
    - https://arxiv.org/pdf/1911.12473.pdf -
    maxima achieved at [0]*(d-1) with the 'largest' categorical value
    Args:
        X: m x (d+C) tensor with the first C columns indicate one-hot encoded categorical variables
        C: number of categories
    Returns:
        y: m x 1 tensor of function valuation
    '''
    m = X.shape[0]
    d = X.shape[-1] - C
    
    c = torch.argmax(X[:, :C], dim=-1, keepdims=True)
    c = C - c - 1
    Z = X[..., 1:] + c
    
    first_term = -20 * torch.exp(-0.2 * torch.sqrt(1/(d) * (Z**2).sum(dim=-1, keepdim=True)))
    second_term = - torch.exp(1/(d) * (torch.cos(2*np.pi*Z)).sum(dim=-1, keepdim=True))
    
    y = -(first_term + second_term + 20 + np.exp(1) + c/2)
        
    return y


def generate_X(m: int, C:int, bounds: torch.Tensor, seed=None) -> torch.Tensor:
    '''
    generate random samples with sobol sequencing with first C column being one-hot-coded categories
    Args:
        m: number of datapoints per category
        C: number of categories
        bounds: 2 x d tensor with first row indicating lower bounds and 2nd row indicating upper bounds. 
    Returns:
        X: m x (d+C) tensor with first C column being one-hot-coded categorical variables: {0, 1}
    '''
    torch.manual_seed(seed)
    seeds = torch.randint(0, 10000, (C,))
    
    X_all = []
    for i in range(C):
        X = draw_sobol_samples(bounds, 1, m, seed=seeds[i].item()).squeeze(0)
        cat_features = torch.zeros(m, C).to(X)
        cat_features[:, i] = 1.0
        X = torch.cat([cat_features, X], dim=-1)
        X_all.append(X)
    
    X_all = torch.cat(X_all, dim=0)
    return X_all


def normalize_with_bounds(X: torch.Tensor, C: int, bounds:torch.Tensor):
    
    assert X.shape[-1]-C == bounds.shape[-1], 'X and bounds shapes are mismatched'
    categories = X[..., 0:C]
    lower = bounds[0:1, :]
    upper = bounds[1:2, :]
    X_normalized = (X[:, C:] - lower) / (upper - lower)
    X_normalized = torch.cat([categories, X_normalized], dim=-1)
    return X_normalized


def denormalize_with_bounds(X: torch.Tensor, C: int, bounds: torch.Tensor):
    
    assert X.shape[-1]-C == bounds.shape[-1], 'X and bounds shapes are mismatched'
    categories = X[..., 0:C]
    lower = bounds[0:1, :]
    upper = bounds[1:2, :]
    X_denormalized = X[:, C:] * (upper - lower) + lower
    X_denormalized = torch.cat([categories, X_denormalized], dim=-1)
    return X_denormalized


def initialize_models(train_X: torch.Tensor, 
                      train_y: torch.Tensor,
                      nu: float=2.5):
    '''
    initialize GP model.
    Args:
        train_X: training inputs
        train_y: training outputs
        nu: lengthscale
    Returns:
        BoTorch model
    '''
    dim_X = train_X.shape[-1]
    dim_y = train_y.shape[-1]
    
    # covar_module handling
    kernel_kwargs = {"nu": nu, "ard_num_dims": dim_X}
    base_kernel = MaternKernel(**kernel_kwargs)
    covar_module = ScaleKernel(base_kernel)
    
    model = SingleTaskGP(train_X, 
                         train_y, 
                         covar_module=covar_module, 
                         outcome_transform=Standardize(dim_y)).to(train_X)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={'disp': False, 'maxiter': 500})
    
    return model


def get_candidate(acqf,
                  bounds: torch.Tensor,
                  C: int) -> torch.Tensor:
    '''
    optimize acqf and get K candidates
    procedure: for each category, optimize acqf, pool candidates across all categories select K best based on their acqf valuation
    Args:
        acqf: acquisition function defined with BoTorch
        bounds: tensor bounds for acqf
        C:  number of categories
    Returns:
        1 x (d+C) tensor with d = dim of continuous variables, C = number of categories
    '''
    # defining list of dictionaries of fixed values defined for optimize_acqf
    fixed_values_list = []
    for i in range(C):
        list_for_dict = [[x, 0] for x in range(C)]
        list_for_dict[i][-1] = 1.0
        fixed_values = {x: y for x, y in list_for_dict}
        fixed_values_list.append(fixed_values)
        
    candidates_per_cat = []
    for i in range(C):
        candidate, _ = optimize_acqf(acq_function=acqf,
                                     bounds=bounds,
                                     q=1,
                                     num_restarts=20,
                                     raw_samples=512,
                                     fixed_features=fixed_values_list[i])
        candidates_per_cat.append(candidate)
    
    candidates_per_cat = torch.cat(candidates_per_cat).unsqueeze(1)
    idx_max = torch.argmax(acqf(candidates_per_cat))
    return candidates_per_cat[idx_max]
                                     
                                    
                                
                   
    

