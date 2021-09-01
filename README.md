# Multi-Armed Bandit - Bayesian Optimization (TODO: Update!)

This project is to explore Multi-Armed Bandit Bayesian Optimization as a method to optimize black-box functions with mixed categorical and continuous variables.

This work is based on a research paper by Dang Nguyen, Sunil Gupta, Santu Rana, Alistair Shilton, Svetha Venkatesh - Applied Artificial Intelligence Institute (<img src="https://render.githubusercontent.com/render/math?math=A^2I^2">), Deakin University, Geelong, Australia - {d.nguyen, sunil.gupta, santu.rana, alistair.shilton, svetha.venkatesh}@deakin.edu.au

The Bayesian Optimization used is BoTorch for Python. Notebooks can be found in 'notebooks' folder.

Proposed MAB-BO has the following algorithm for batched observations:

![image](https://user-images.githubusercontent.com/58320929/123740893-be797380-d8db-11eb-90cd-f684aaa4dd9e.png)

Tests are performed on negated modified Ackley function based on the same paper with minor modification to make the objective values shift between different categories smaller and having the category effect reversed. Global optima is 0 at [0]\*d where d is the number of dimension and is achieved with category c = c_max with c ∈ {0, 1, 2, ..., c_max}


### Test 1 - vs One-Hot encoding (OH-BO)

General params:
* Ackley domain: [-1]\*d to [3]\*d
* Surrogate model: Gaussian Process
* Covariance kernel: Matern (nu = 0.5)
* d: 5 (dimension of continuous variables)
* \# of categories: 5
* \# of observations per iteration: 1
* \# of iterations: 100
* \# of trials: 5

MAB-BO params:
* \# of Thompson samples: 5000
* sampling method: cholesky

OH-BO params:
* acquisition function: qEI

Results:

![image](https://user-images.githubusercontent.com/58320929/131598294-d4b8c686-6656-49e0-81c5-bbe51ba7d96d.png)


### Test 2 - vs One-Hot encoding (OH-BO) with wider domain

General params:
* Ackley domain: [-10]\*d to [10]\*d
* Surrogate model: Gaussian Process
* Covariance kernel: Matern (nu = 0.5)
* d: 5 (dimension of continuous variables)
* \# of categories: 5
* \# of observations per iteration: 1
* \# of iterations: 100
* \# of trials: 5

MAB-BO params:
* \# of Thompson samples: 5000
* sampling method: cholesky

OH-BO params:
* acquisition function: qEI

Results:

![image](https://user-images.githubusercontent.com/58320929/131598403-1f82af6e-7e5d-489a-812f-b48101f74128.png)

