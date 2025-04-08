import torch as th
from pdb import set_trace as bp
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as fun
from scipy.io import loadmat
import math
import copy
import pickle as pkl
import matplotlib.pyplot as plt
from torchsummary import summary
import time
import torch.optim.lr_scheduler as lr_scheduler
from collections import deque
import random

th.autograd.set_detect_anomaly(True)

cuda = True if th.cuda.is_available() else False
device = th.device("cuda" if cuda else "cpu")

total_simu_times = 2000
scale_h = 1
scale_g = 1
sigma_n = 0.31
sigma_v = 0.31
epsilon_small = 0.8 ### create a stringent baseline feasible condition
epsilon = 1 ### real epsilon
ave_rate = 0.5
quan_step = 0.01
power_step = 0.01


### check the infeasibility of the rate allocation
def infeasible_check(h, g, sigma_n, sigma_v, epsilon, rate_require):
    up_power_sum = np.log1p((epsilon/sigma_n)*(h/g)).sum()
    flag = True
    L = len(h)
    for l in range(L):
        if h[l]/sigma_n >= g[l]/sigma_v and up_power_sum >= rate_require:
            flag = False
            break
    return flag

#### box constraint
def rate_require_given_lambda(lam, beta_allo_up, h_samples):
    temp = np.maximum(0.0, np.log((lam*h_samples)/sigma_n))
    temp = np.minimum(temp, beta_allo_up)
    return np.sum(temp)


def solve_rate_allocation(h, g, rate_require, epsilon, sigma_n, tol=1e-9, max_iter=1000):
    h = np.array(h, dtype=float)
    g = np.array(g, dtype=float)
    L = len(h)

    beta_ub = np.log1p((epsilon/sigma_n)*(h/g)) 
    # if np.sum(beta_ub) <= rate_require:
    #     return beta_ub
    
    ### else: P_l = min( epsilon/g_l , max(0, eta - sigma_n/h_l ) )
    left = 0.0
    right = np.max(beta_ub) + 1.0 ## a small margin

    ### binary search
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        current_sum = rate_require_given_lambda(mid, beta_ub, h)

        if abs(current_sum - rate_require) < tol:
            break
        if current_sum > rate_require:
            right = mid
        else:
            left = mid

    ## obtain the optimal power allo based on converged eta
    lam_star = 0.5 * (left + right)
    beta_opt = np.maximum(0.0, np.log((lam_star*h)/sigma_n))
    beta_opt = np.minimum(beta_opt, beta_ub)
    power_opt = ((np.exp(beta_opt)-1)*(sigma_n/h)).sum()

    return beta_opt, power_opt


### less noisy constraint calculation
def delta_func(beta, h, g, sigma_n, sigma_v):
    term1 = np.sum(beta)   # log(1 + P_l*h_l/sigma_n)
    term2 = np.log1p((np.exp(beta)-1)*(sigma_n*g)/(sigma_v*h)).sum()  # log(1 + P_l*g_l/sigma_v)
    return term1 - term2

## \partial \mathcal{G} / \partial \beta_\ell
def grad_objective(beta, h, g, sigma_n, sigma_v, eta, b):
    prop = (sigma_n * g) / (sigma_v * h)
    delta_temp = delta_func(beta, h, g, sigma_n, sigma_v)
    grad = np.exp(beta) * (sigma_n / h) + 2.0 * eta * (delta_temp - b) * ((1-prop)/(1+(np.exp(beta)-1)*prop))
    return grad ## vector

## \partial \mathcal{G} / \partial b
def grad_b(beta, h, g, sigma_n, sigma_v, eta, b):
    delta_temp = delta_func(beta, h, g, sigma_n, sigma_v)
    grad = 2.0 * eta * (b - delta_temp)
    return grad

## iteratively project P onto the feasible set
def project_onto_beta(beta, h, g, rate_require, epsilon, pocs_max_iter=100, pocs_tol=1e-9):
    beta_proj = beta.copy()
    upper_bound = np.log1p((epsilon/sigma_n)*(h/g))
    for _ in range(pocs_max_iter):
        beta_prev = beta_proj.copy()

        ## project to 0 <= \beta_\ell <= ...)
        beta_proj = np.minimum(beta_proj, upper_bound)
        beta_proj = np.maximum(beta_proj, 0.0)

        ## project to sum_\ell \beta_\ell >= rate_require
        s = beta_proj.sum()
        if s < rate_require:
            beta_proj *= (rate_require / s)

        # convergence check
        if np.linalg.norm(beta_proj - beta_prev) < pocs_tol:
            break

    return beta_proj



#### less stable, but faster convergence rates
def projected_gradient_descent_with_penalty(
        h, g,
        beta_init,      ## water-filling solution
        sigma_n, sigma_v,
        rate_require, epsilon,
        alpha_beta, alpha_b,   # step size of P and b updates
        gamma, C,  # penalty parameters                  
        max_iter, tol
    ):

    beta = beta_init.copy()
    b = 0.0  
    L = len(beta)

    D0 = delta_func(beta, h, g, sigma_n, sigma_v)
    sum_exp = ((np.exp(beta)-1)*(sigma_n/h)).sum()
    # if D0 < 0.0:
    #     eta0 = (C / (D0**2)) * max(sum_exp, 1e-6)  ## prevent sigular value
    # else:
    #     eta0 = 1e-6

    eta0 = (C / (D0**2)) * max(sum_exp, 1e-6)  ## prevent sigular value

    history_power = []
    history_delta = []
    D_val_max = -1e8
    for n in range(max_iter):
        # stop_time = n % 1
        # if stop_time == 0:
        #     eta_n = eta0 * (1.0 + gamma*(stop_time))
        eta_n = eta0 * (1.0 + gamma*n)

        ## gradient descent of beta
        gradient_beta = grad_objective(beta, h, g, sigma_n, sigma_v, eta_n, b)
        beta_tmp = beta - alpha_beta * gradient_beta  ## decrease, should be in negative direction

        ## project
        beta_new = project_onto_beta(beta_tmp, h, g, rate_require, epsilon, pocs_max_iter=100, pocs_tol=1e-9)

        ## gradient descent of b
        gb = grad_b(beta_new, h, g, sigma_n, sigma_v, eta_n, b)
        b_new = b - alpha_b * gb
        b_new = max(b_new, 0.0)

        ## convergence conditions: dP, db, Dval
        dbeta = np.linalg.norm(beta_new - beta)
        db = abs(b_new - b)
        Dval = delta_func(beta_new, h, g, sigma_n, sigma_v)
        D_val_max = max(D_val_max, Dval)
        
        # update
        beta, b = beta_new, b_new
        power_new = ((np.exp(beta)-1)*(sigma_n/h)).sum()

        history_power.append(power_new)
        history_delta.append(Dval)

        # if n % 10 == 0:
        #     print(f"Iteration {n}: Power = {power_new}, Delta = {Dval}, dP = {dbeta}, db = {db}")

        ## stopping criteria
        if dbeta < tol and db < tol and Dval >= 0:  # Dval >= -1e-9 to account for numerical errors
            # print(f'Converged at iteration {n}: Power = {power_new}, Delta = {Dval}, dP = {dbeta}, db = {db}')
            # print("feasible!")
            break
    power_final = power_new
    sum_rate_final = beta.sum()
    ### sometimes we can only determine the feasibility in the end
    # if Dval < -1e-4:
        # print("perhaps infeasible based on this algorithm!")

    return beta, b, history_power, history_delta, power_final, sum_rate_final, D_val_max



##### more stable, but slower convergence rates, choose to use which one
def projected_gradient_descent_with_penalty_stable(
        h, g,
        beta_init,      # initial solution (e.g., water-filling solution or another proper initialization)
        sigma_n, sigma_v,
        rate_require, epsilon,
        alpha_beta_init, alpha_b_init,   # initial step sizes for beta and b updates
        gamma, C,  # penalty parameters                  
        max_iter, tol
    ):
    """
    Improved projected gradient descent algorithm with a penalty function:
      - Adaptive decaying step sizes,
      - Gradient clipping,
      - Log-scheduled update for the penalty parameter.
    Returns:
      beta: optimized power allocation parameters (in log domain)
      b: auxiliary variable
      history_power: history of total power values at each iteration
      history_delta: history of delta (feasibility metric) at each iteration
      power_final: final total power
      sum_rate_final: final total rate (sum of beta values)
      D_val_max: maximum delta observed during iterations
    """
    beta = beta_init.copy()
    b = 0.0  
    L = len(beta)

    # Compute the initial penalty factor eta0; ensure non-zero value to avoid division-by-zero issues.
    D0 = delta_func(beta, h, g, sigma_n, sigma_v)
    sum_exp = ((np.exp(beta) - 1) * (sigma_n / h)).sum()
    eta0 = (C / (D0**2)) * max(sum_exp, 1e-6)
    
    # Set a gradient clipping threshold to prevent excessively large updates in one step.
    grad_clip_threshold = 100.0

    history_power = []
    history_delta = []
    D_val_max = -1e8

    # Initialize the step sizes.
    alpha_beta = alpha_beta_init
    alpha_b = alpha_b_init

    for n in range(max_iter):
        # Adaptive step sizes: reduce the update magnitude as the number of iterations increases.
        alpha_beta_n = alpha_beta / (1 + n / 1000.0)
        alpha_b_n = alpha_b / (1 + n / 1000.0)

        # Update the penalty parameter eta using a logarithmic schedule for a smoother increase.
        eta_n = eta0 * (1.0 + gamma * np.log(n + 1))

        # Compute the gradient for beta and perform gradient clipping.
        gradient_beta = grad_objective(beta, h, g, sigma_n, sigma_v, eta_n, b)
        grad_norm = np.linalg.norm(gradient_beta)
        if grad_norm > grad_clip_threshold:
            gradient_beta = gradient_beta * (grad_clip_threshold / grad_norm)
        
        # Update beta using the computed gradient and then project beta onto the feasible domain.
        beta_tmp = beta - alpha_beta_n * gradient_beta  
        beta_new = project_onto_beta(beta_tmp, h, g, rate_require, epsilon, pocs_max_iter=100, pocs_tol=1e-9)

        # Compute the gradient for the auxiliary variable b and clip it if necessary.
        gb = grad_b(beta_new, h, g, sigma_n, sigma_v, eta_n, b)
        gb_norm = abs(gb)
        if gb_norm > grad_clip_threshold:
            gb = gb * (grad_clip_threshold / gb_norm)
        b_new = b - alpha_b_n * gb
        b_new = max(b_new, 0.0)  # b is maintained as non-negative

        # Compute convergence metrics: change in beta, change in b, and the delta value.
        dbeta = np.linalg.norm(beta_new - beta)
        db = abs(b_new - b)
        Dval = delta_func(beta_new, h, g, sigma_n, sigma_v)
        D_val_max = max(D_val_max, Dval)
        
        # Record the total power and delta value at this iteration to track progress.
        power_new = ((np.exp(beta_new) - 1) * (sigma_n / h)).sum()
        history_power.append(power_new)
        history_delta.append(Dval)

        # Update variables for the next iteration.
        beta, b = beta_new, b_new

        # Stopping condition: if both beta and b updates are smaller than tol and delta indicates feasibility (>=0), break.
        if dbeta < tol and db < tol and Dval >= 0:
            break

    power_final = ((np.exp(beta) - 1) * (sigma_n / h)).sum()
    sum_rate_final = beta.sum()
    return beta, b, history_power, history_delta, power_final, sum_rate_final, D_val_max



##### baseline_solution_ver_1
# def baseline_nonconvex_solution(h,g,rate_require,sigma_n,sigma_v,epsilon):
#     L = len(h)
#     beta = np.zeros(L)
#     bound_prop = 0
#     for l in range(L):
#         if h[l]/sigma_n >= g[l]/sigma_v:
#             beta_bound = np.log1p((epsilon/sigma_n)*(h[l]/g[l]))
#             bound_prop += beta_bound
#             beta[l] = beta_bound
#     if bound_prop > rate_require:
#         beta = beta * (rate_require / bound_prop)
#     sum_power = ((np.exp(beta)-1)*(sigma_n/h)).sum()
#     # return beta, sum_power
#     sum_beta = sum(beta)
#     return beta, sum_beta, sum_power


##### baseline_solution_ver_2
def sum_rate_alpha_power(h,g,sigma_n,sigma_v,epsilon,alpha):
    L = len(h)
    allo_power = np.zeros(L)
    allo_rate = np.zeros(L)
    for l in range(L):
        if h[l]/sigma_n >= g[l]/sigma_v:
            allo_power[l] = (alpha*epsilon) / g[l]
            allo_rate[l] = np.log1p((allo_power[l]*h[l])/sigma_n)
    sum_rate = allo_rate.sum()
    sum_power = allo_power.sum()
    return sum_rate, allo_rate, sum_power

def baseline_nonconvex_solution(h,g,rate_require,sigma_n,sigma_v,epsilon,max_iter=1000, tol=1e-6):  
    ## the range of alpha is [0, 1]
    left = 0.0
    right = 1.0 

    sum_rate_ub,_,_ = sum_rate_alpha_power(h,g,sigma_n,sigma_v,epsilon,right) 
    if sum_rate_ub < rate_require:
        feasi_flag = False
        beta_base = np.zeros(len(h))
        sum_rate_base = 0
        sum_power_base = 0
    else: 
        feasi_flag = True
        ### binary search of alpha
        for _ in range(max_iter):
            mid = 0.5 * (left + right)
            current_sum,_,_ = sum_rate_alpha_power(h,g,sigma_n,sigma_v,epsilon,mid)

            if abs(current_sum - rate_require) < tol:
                break
            if current_sum > rate_require:
                right = mid
            else:
                left = mid
        ## obtain the optimal power allo based on converged eta
        alpha_star = 0.5 * (left + right)
        # print(f"alpha_star = {alpha_star}")
        sum_rate_base, beta_base, sum_power_base = sum_rate_alpha_power(h,g,sigma_n,sigma_v,epsilon,alpha_star)
    
    return feasi_flag, sum_rate_base, beta_base, sum_power_base


#### generate discrete channel gains
h_value_range = th.arange(quan_step,10,quan_step)
ray_h_value = (th.exp(-(h_value_range)/(2*scale_h)))/(2*scale_h)
norm_h_value = sum(ray_h_value)
prob_h = ray_h_value/norm_h_value ## after normalization

g_value_range = th.arange(quan_step,10,quan_step)
ray_g_value = (th.exp(-(g_value_range)/(2*scale_g)))/(2*scale_g)
norm_g_value = sum(ray_g_value)
prob_g = ray_g_value/norm_g_value ## after normalization


###### Monte Carlo simulation #####
total_block = 10
rate_require = total_block * ave_rate
infeasible_times = 0
good_chan_times = 0
bad_chan_times = 0

pgd_feasi_times = 0
baseline_feasi_times = 0
pgd_infeasible_times = 0
baseline_infeasible_times = 0
pgd_baseline_infeasi_times = 0
pgd_baseline_feasi_times = 0
trivial_feasible_times = 0

bad_chan_total_power_pgd = 0
bad_chan_total_power_baseline = 0
total_power_pgd = 0
total_power_baseline = 0
total_power_trivial = 0
total_rate_pgd = 0
total_rate_baseline = 0
total_rate_trivial = 0

alpha_rate = 1e-6
max_iter = 3000 
C_init = 20.0
gamma = 0.2

## generate discrete channel gains
h_value_range = th.arange(quan_step,10,quan_step)
ray_h_value = (th.exp(-(h_value_range)/(2*scale_h)))/(2*scale_h)
norm_h_value = sum(ray_h_value)
prob_h = ray_h_value/norm_h_value ## after normalization

g_value_range = th.arange(quan_step,10,quan_step)
ray_g_value = (th.exp(-(g_value_range)/(2*scale_g)))/(2*scale_g)
norm_g_value = sum(ray_g_value)
prob_g = ray_g_value/norm_g_value ## after normalization


##### single test #####

## pass two infeasibility checks, convex solution infeasible, 
flag = True
while flag == True or feasi_flag == False or gap >= 0:

    h_samples = h_value_range[th.multinomial(prob_h, total_block, replacement=True)] 
    g_samples = g_value_range[th.multinomial(prob_g, total_block, replacement=True)]
    th.save(h_samples, 'h_samples_rate.pth')
    th.save(g_samples, 'g_samples_rate.pth')

    flag = infeasible_check(h_samples, g_samples, sigma_n, sigma_v, epsilon, rate_require)
    if flag:
        # print("case 1: infeasible!")
        continue
    ## if two infeasibility conditions do not hold, we can go on
    else:
        beta_wf, power_wf = solve_rate_allocation(h_samples, g_samples, rate_require, epsilon, sigma_n, tol=1e-9, max_iter=1000)
        # print(beta_wf)
        # print(np.sum(power_wf))

        h_samples = h_samples.numpy()
        g_samples = g_samples.numpy()
        feasi_flag, _, beta_base, _ = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon,max_iter=1000, tol=1e-9)
        ## whether satisfying the less noisy constraint
        gap = delta_func(beta_wf, h_samples, g_samples, sigma_n, sigma_v)

# h_samples = th.load('h_samples_rate.pth')
# g_samples = th.load('g_samples_rate.pth')
h_samples = th.load('h_samples_rate_1.pth')
g_samples = th.load('g_samples_rate_1.pth')
beta_wf, _ = solve_rate_allocation(h_samples, g_samples, rate_require, epsilon, sigma_n, tol=1e-9, max_iter=1000)
h_samples = h_samples.numpy()
g_samples = g_samples.numpy()
feasi_flag, _, beta_base, _ = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon,max_iter=1000, tol=1e-9)

random_values = np.random.rand(total_block)
normalized_random_values = random_values / np.sum(random_values)
beta_rand = rate_require * normalized_random_values

# if gap >= 0 or feasi_flag == False:
#     beta_opt_allo = beta_wf
#     print(f"case 2: convex feasible!")
#     print(gap)
#     print(feasi_flag)
# else:  ### currently "alpha = 1e-6, gamma = 0.2, C = 20.0" is the best

#### initialized with inverse water-filling solution
beta_opt_allo, b_opt, hist_power, hist_delta, _, _, _ = projected_gradient_descent_with_penalty(
    h_samples, g_samples, beta_wf,
    sigma_n, sigma_v,
    rate_require, epsilon,
    alpha_beta=alpha_rate, alpha_b=alpha_rate,
    gamma=gamma, C=C_init,
    max_iter=max_iter, tol=1e-6
)
# bit_grade = sum(beta_opt_allo)
# print(f"sum rate = {bit_grade}")

#### initialized with trivial solution
beta_opt_allo_2, b_opt_2, hist_power_2, hist_delta_2, _, _, _ = projected_gradient_descent_with_penalty_stable(
    h_samples, g_samples, beta_base,
    sigma_n, sigma_v,
    rate_require, epsilon,
    # alpha_beta=alpha_rate, alpha_b=alpha_rate,
    alpha_beta_init=alpha_rate, alpha_b_init=alpha_rate,
    gamma=gamma, C=C_init,
    max_iter=max_iter, tol=1e-6
)

#### initialized with random solution
beta_opt_allo_3, b_opt_3, hist_power_3, hist_delta_3, _, _, _ = projected_gradient_descent_with_penalty_stable(
    h_samples, g_samples, beta_rand,
    sigma_n, sigma_v,
    rate_require, epsilon,
    # alpha_beta=alpha_rate, alpha_b=alpha_rate,
    alpha_beta_init=alpha_rate, alpha_b_init=alpha_rate,
    gamma=gamma, C=C_init,
    max_iter=max_iter, tol=1e-6
)

def pad_sequence(seq, max_length):
    return np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=np.nan)

max_length = max(len(hist_power), len(hist_power_2), len(hist_power_3))
hist_power = pad_sequence(hist_power, max_length)
hist_power_2 = pad_sequence(hist_power_2, max_length)
hist_power_3 = pad_sequence(hist_power_3, max_length)

max_length_delta = max(len(hist_delta), len(hist_delta_2), len(hist_delta_3))
hist_delta = pad_sequence(hist_delta, max_length_delta)
hist_delta_2 = pad_sequence(hist_delta_2, max_length_delta)
hist_delta_3 = pad_sequence(hist_delta_3, max_length_delta)

iterations = range(len(hist_power))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(iterations, hist_power, marker='o', linestyle='-', color='b', label='Power, convex initialization')
plt.plot(iterations, hist_power_2, marker='*', linestyle='-', color='r', label='Power, trivial initialization')
plt.plot(iterations, hist_power_3, marker='^', linestyle='-', color='g', label='Power, random initialization')
# plt.xlabel('Iteration n')
plt.xlabel('Update step k')
plt.ylabel('Power (dB)')
plt.title('Total power consumption')
plt.grid(True, which="both", ls="--")
plt.legend()

interval = 600
plt.xticks(np.arange(0, len(iterations), interval), labels=[f'{i//30}' for i in np.arange(0, len(iterations), interval)])

plt.subplot(1, 2, 2)
plt.plot(iterations, hist_delta, marker='o', linestyle='-', color='b', label='Delta, convex initialization')
plt.plot(iterations, hist_delta_2, marker='*', linestyle='-', color='r', label='Delta, trivial initialization')
plt.plot(iterations, hist_delta_3, marker='^', linestyle='-', color='g', label='Delta, random initialization')
# plt.xlabel('Iteration n')
plt.xlabel('Update step k')
# plt.ylabel('Delta')
plt.title('Feasibility')
plt.grid(True, which="both", ls="--")
plt.legend()

interval = 600
plt.xticks(np.arange(0, len(iterations), interval), labels=[f'{i//30}' for i in np.arange(0, len(iterations), interval)])

plt.tight_layout()
plt.show()

# print(f"optimal power allocation: {beta_opt_allo}, sum = {np.sum(beta_opt_allo)}")

# plt.figure(figsize=(10, 6))
# plt.bar(range(len(beta_opt_allo)), beta_opt_allo, color='blue')
# plt.xlabel('Index')
# plt.ylabel('Power Level')
# plt.title('Power Allocation')
# plt.show()
