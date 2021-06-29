from contextlib import ExitStack

import numpy as np

import torch
tkwargs = {"dtype": torch.double,
#            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
           "device": torch.device("cpu")
          }
from torch.quasirandom import SobolEngine

from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.generation.sampling import MaxPosteriorSampling

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
import gpytorch.settings as gpts

from ModifiedSingleTaskGP import ModifiedSingleTaskGP


# modified (and negated) Ackley-5d function
def modified_neg_ackley(X: torch.Tensor) -> torch.Tensor:
    '''
    compute modified ackley function as in:
    "Bayesian Optimization for Categorical and Category-Specific Continuous Inputs"
    - Dang Nguyen, Sunil Gupta, Santu Rana, Alistair Shilton, Svetha Venkatesh -
    - https://arxiv.org/pdf/1911.12473.pdf -
    maxima achieved at [0]*(d-1) with the 'largest' categorical value
    Args:
        X: m x (d+1) tensor with first column hosting categorical variable
    Returns:
        y: m x 1 tensor of function valuation
    '''
    m, d = X.shape
    d -= 1    # only the continuous variables
    c = torch.round(X[..., 0:1])
    c = c.max() - c - 1    # reversing category
    Z = X[..., 1:] + c
    
    first_term = -20 * torch.exp(-0.2 * torch.sqrt(1/(d) * (Z**2).sum(dim=-1, keepdim=True)))
    second_term = - torch.exp(1/(d) * (torch.cos(2*np.pi*Z)).sum(dim=-1, keepdim=True))
    
    y = -(first_term + second_term + 20 + np.exp(1) + c/2)
        
    return y


def generate_X(m: int, C: int, bounds: torch.Tensor, seed=None) -> torch.Tensor:
    '''
    generate random samples with sobol sequencing with first column being categorial variable
    Args:
        m: number of datapoints per category
        C: number of categories
        bounds: 2 x d tensor with first row indicating lower bounds and 2nd row indicating upper bounds.
    Returns:
        X: m x (d+1) tensor with first column being categorical variable: {0, 1, ..., c-1}
    '''
    torch.manual_seed(seed)
    seeds = torch.randint(0, 10000, (C,))
    
    X_all = []
    for i in range(C):
        X = draw_sobol_samples(bounds, 1, m, seed=seeds[i].item()).squeeze(0)
        categories = torch.tensor([i]*m).unsqueeze(-1).to(X)
        X = torch.cat([categories, X], dim=-1)
        X_all.append(X)
    
    X_all = torch.cat(X_all, dim=0)
    return X_all


def normalize_X(X: torch.Tensor):
    
    d = X.shape[-1] - 1
    categories = X[..., 0:1]
    C = categories.to(int).max().item() + 1
    
    X_normalized = []    # to store normalized X (without categories)
    normalizers = []    # to store normalizer for each category
    for c in range(C):
        X_c = X[X[..., 0] == c, 1:]
        assert X_c.shape[-2] > 0, 'empty category'
        normalizer = Normalize(d=d)
        X_c_normalized = normalizer._transform(X_c)
        X_normalized.append(X_c_normalized)
        normalizers.append(normalizer)
        
    X_normalized = torch.cat(X_normalized, dim=0)
    X_normalized = torch.cat([categories, X_normalized], dim=-1)
    
    return X_normalized, normalizers



def denormalize_X(X: torch.Tensor, normalizers):
    
    d = X.shape[-1] - 1
    categories = X[..., 0:1]
    max_c = categories.to(int).max().item()
    C = len(normalizers)
    
    assert max_c < C, 'category out of range'
    
    X_denormalized = []
    for c in range(C):
        X_c = X[X[..., 0] == c, 1:]
        if X_c.shape[-2] == 0: continue
        X_c_denormalized = normalizers[c]._untransform(X_c)
        X_denormalized.append(X_c_denormalized)
    
    X_denormalized = torch.cat(X_denormalized, dim=0)
    X_denormalized = torch.cat([categories, X_denormalized], dim=-1)
    
    return X_denormalized



def normalize_with_bounds(X: torch.Tensor, bounds:torch.Tensor):
    
    assert X.shape[-1]-1 == bounds.shape[-1], 'X and bounds shapes are mismatched'
    categories = X[..., 0:1]
    lower = bounds[0:1, :]
    upper = bounds[1:2, :]
    X_normalized = (X[:, 1:] - lower) / (upper - lower)
    X_normalized = torch.cat([categories, X_normalized], dim=-1)
    return X_normalized


def denormalize_with_bounds(X: torch.Tensor, bounds: torch.Tensor):
    
    assert X.shape[-1]-1 == bounds.shape[-1], 'X and bounds shapes are mismatched'
    categories = X[..., 0:1]
    lower = bounds[0:1, :]
    upper = bounds[1:2, :]
    X_denormalized = X[:, 1:] * (upper - lower) + lower
    X_denormalized = torch.cat([categories, X_denormalized], dim=-1)
    return X_denormalized



def initialize_models(train_X: torch.Tensor, 
                      train_y: torch.Tensor,
                      nu: float=2.5,
                      sampler='cholesky', 
                      use_keops:bool=False) -> tuple:
    '''
    initialize model list (list of c models with c = number of categories)
    Args:
        train_X: m x d tensor
        train_y: m x 1 tensor
        sampler: sampler: "cholesky", "ciq", "rff" or "lanczos"
    Returns:
        list of models
    '''
    # covar_module handling
    kernel_kwargs = {"nu": nu, "ard_num_dims": train_X.shape[-1]-1}
    if sampler == "rff":
        base_kernel = RFFKernel(**kernel_kwargs, num_samples=1024)
    else:
        base_kernel = (
            KMaternKernel(**kernel_kwargs) if use_keops else MaternKernel(**kernel_kwargs)
        )
    covar_module = ScaleKernel(base_kernel)
    
    d = train_X.shape[-1] - 1
    C = train_X[..., 0].to(int).max().item() + 1
    
    models = []
    mlls = []
    for i in range(C):
        mask = torch.round(train_X[..., 0]) == i
        X, y = train_X[mask], train_y[mask]
        assert X.shape[0] >= 2*(d+1), f"not enough datapoints for category {i}"
        model = SingleTaskGP(X[..., 1:], y, covar_module=covar_module, outcome_transform=None).to(**tkwargs)
        mlls.append(ExactMarginalLogLikelihood(model.likelihood, model))
        models.append(model)
    
    for mll in mlls:
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={'disp': False, 'maxiter': 500})
        
    return models



def thompson_sampling(model, d:int, n_samples: int, sampler='cholesky') -> float:
    '''
    samples n_candidates with sobol and the corresponding posterior sample and return the max objective value
    Args:
        model: BoTorch GP model
        n_samples: number of candidates X sampled
        d: dimension in the model
        sampler: "cholesky", "ciq", "rff" or "lanczos"
    Returns:
        4-tuple: max posterior-sampled value, x corresponding to said max (1 x d tensor), 
                 x sampled (n_samples x d tensor), samples posterior values (n_samples x 1 tensor)
    '''
    assert sampler in ("cholesky", "ciq", "rff", "lanczos")
    
    # sampling candidates
    sobol = SobolEngine(d, scramble=True)
    X_cand = sobol.draw(n_samples).to(**tkwargs)
    
    # Thompson sampling setting
    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(gpts.minres_tolerance(2e-3))  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
  
    # sampling posterior
    posterior = model.posterior(X_cand)
    samples = posterior.rsample().squeeze(0).detach()
    idx_max = torch.argmax(samples)
    
    return samples[idx_max].item(),  X_cand[idx_max: idx_max+1, :], X_cand, samples



def pick_category(models, d: int, n_samples: int, sampler='cholesky') -> int:
    '''
    run thompson sampling for each model in models and pick the category corresponding to max value of sample
    Args:
        models: BoTorch ModelListGP object
        n_samples: number of candidates X sampled
        d: dimension in the model
        sampler: "cholesky", "ciq", "rff" or "lanczos"
    Returns:
        category number from {0, 1, ..., len(models)-1}
    '''
    thompson_samples = []
    for model in models:
        sample = thompson_sampling(model, d, n_samples, sampler=sampler)[0]
        thompson_samples.append(sample)
    
    thompson_samples = torch.tensor(thompson_samples)
    return torch.argmax(thompson_samples).item()


def pick_category_and_candidate(models, d: int, n_samples: int, sampler='cholesky') -> int:
    '''
    run thompson sampling for each model in models and pick the category and candidate corresponding to max value of sample
    Args:
        models: BoTorch ModelListGP object
        n_samples: number of candidates X sampled
        d: dimension in the model
        sampler: "cholesky", "ciq", "rff" or "lanczos"
    Returns:
        tuple: category number from {0, 1, ..., len(models)-1}, candidate X (1 x d tensor)
    '''
    thompson_samples = []
    candidates = []
    for model in models:
        sample, candidate, _, _ = thompson_sampling(model, d, n_samples, sampler=sampler)
        thompson_samples.append(sample)
        candidates.append(candidate)
    
    thompson_samples = torch.tensor(thompson_samples)
    best_category = torch.argmax(thompson_samples).item()
    best_candidate = candidates[best_category]
    return best_category, best_candidate



def get_candidates(models, 
                   d: int,
                   n_candidates: int, 
                   n_samples: int,
                   sampler='cholesky') -> torch.Tensor:
    '''
    get candidate from category picked from 'pick_category' function based on max posterior.
    Args:
        models: BoTorch ModelListGP object
        n_candidates: number of candidates generated
        n_samples: number of posterior samples
        d: dimension in the model
        sampler: "cholesky", "ciq", "rff" or "lanczos"
    Returns:
        candidates: n_candidates x (d+1) tensor
    '''
    candidates = []
    categories = []
    for b in range(n_candidates):
        
#         print(f"generating candidate {b+1}/{n_candidates}")
        c = pick_category(models, d, n_samples, sampler=sampler)
        print(f"picked category: {c} | ", end='')
        categories.append(c)
        
        # generating candidate
        model = models[c]
        sample, candidate, _, _ = thompson_sampling(model, d, n_samples, sampler=sampler)
        print(f"sampled posterior value = {sample}")
        
        
#         # or with MaxPosteriorSampling
#         sobol = SobolEngine(d, scramble=True)
#         X_cand = sobol.draw(n_samples).to(**tkwargs)
#         thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
#         candidate = thompson_sampling(X_cand, num_samples=1)

        candidates.append(candidate)
    
    candidates = torch.cat(candidates, dim=0)
    categories = torch.tensor(categories).unsqueeze(-1).to(**tkwargs)
    candidates = torch.cat([categories, candidates], dim=-1)
    
    return candidates



def get_candidates_v2(models, 
                      d: int,
                      n_candidates: int, 
                      n_samples: int,
                      sampler='cholesky') -> torch.Tensor:
    '''
    get candidate from category picked from 'pick_category' function based on max posterior.
    NOTE: v2: thompson sampling stage is only done once (to pick category and the corresponding best candidate at the same time) 
    Args:
        models: BoTorch ModelListGP object
        n_candidates: number of candidates generated
        n_samples: number of posterior samples
        d: dimension in the model
        sampler: "cholesky", "ciq", "rff" or "lanczos"
    Returns:
        candidates: n_candidates x (d+1) tensor
    '''
    candidates = []
    categories = []
    for b in range(n_candidates):
        
#         print(f"generating candidate {b+1}/{n_candidates}")
        c, candidate = pick_category_and_candidate(models, d, n_samples, sampler=sampler)
        print(f"picked category: {c} | ", end='')
        categories.append(c)

        candidates.append(candidate)
    
    candidates = torch.cat(candidates, dim=0)
    categories = torch.tensor(categories).unsqueeze(-1).to(**tkwargs)
    candidates = torch.cat([categories, candidates], dim=-1)
    
    return candidates
        
