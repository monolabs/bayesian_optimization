# Multi-Armed Bandit - Bayesian Optimization

This project is to explore Multi-Armed Bandit Bayesian Optimization as a method to optimize black-box functions with categorical variables.

This work is based on a research paper by Dang Nguyen, Sunil Gupta, Santu Rana, Alistair Shilton, Svetha Venkatesh - Applied Artificial Intelligence Institute (<img src="https://render.githubusercontent.com/render/math?math=A^2I^2">), Deakin University, Geelong, Australia - {d.nguyen, sunil.gupta, santu.rana, alistair.shilton, svetha.venkatesh}@deakin.edu.au

The Bayesian Optimization used is BoTorch for Python. Notebooks can be found in 'notebooks' folder.

Proposed MAB-BO has the following algorithm for batched observations:
![image](https://user-images.githubusercontent.com/58320929/123740893-be797380-d8db-11eb-90cd-f684aaa4dd9e.png)


### Test 1

Surrogate model: Gaussian Process
Covariance kernel: Matern (nu = 0.5)
\# 
