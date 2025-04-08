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

##### parameters
total_block = 10
update_rate = 5 ## target network update freq/ primary network update freq
exp_batch = 100 ## update batch size
train_epochs = 100000

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

#### inverse water-filling rate allocation solution
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

#### PGD for unconstrained optimization
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

        ### linear increase of penalty factor
        eta_n = eta0 * (1.0 + gamma*n)

        ## gradient descent of beta
        gradient_beta = grad_objective(beta, h, g, sigma_n, sigma_v, eta_n, b)
        beta_tmp = beta - alpha_beta * gradient_beta  ## decrease, should be in negative direction

        ## POCS projection
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

        #### if you want to trace the convergence process
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

##### baseline_solution_ver_1: not correct, we are scaling the power allocation
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


##### baseline_solution_ver_2: correct, binary search to find the optimal power scaling
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
max_iter = 2000 
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


##### Monte-Carlo simulations #######
##### Present the average feasibility probabilities and power consumptions (when all schemes are feasible) of different rate allocation schemes 

##### Our proposed method: (1) two infeasiblity checks + (2) water-filling solution and check + (3) PGD and check + (4) trivial solution and check
##### Convex Baseline: (1) two infeasiblity checks + (2) water-filling solution and check + (3) trivial solution and check
##### Trivial Bseline: (1) two infeasiblity checks + (2) trivial solution and check

for epoch in range(total_simu_times):
    print(f"Epoch {epoch}")
    h_samples = h_value_range[th.multinomial(prob_h, total_block, replacement=True)] 
    g_samples = g_value_range[th.multinomial(prob_g, total_block, replacement=True)]

    beta_wf, _ = solve_rate_allocation(h_samples, g_samples, rate_require, epsilon, sigma_n, tol=1e-9, max_iter=1000)
    feasi_flag, sum_rate_trivial_small, _, sum_power_trivial_small = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon)
    if feasi_flag == True:
        trivial_feasible_times += 1
        total_power_trivial += sum_power_trivial_small
        total_rate_trivial += sum_rate_trivial_small

    h_samples = h_samples.numpy()
    g_samples = g_samples.numpy()

    flag = infeasible_check(h_samples, g_samples, sigma_n, sigma_v, epsilon, rate_require)
    gap = delta_func(beta_wf, h_samples, g_samples, sigma_n, sigma_v)

    ### not pass the infeasibility check
    if flag:
        infeasible_times += 1

    ### first stage: convex solution is feasible
    elif gap >= 0:
        good_chan_times += 1
        good_power = ((np.exp(beta_wf)-1)*(sigma_n/h_samples)).sum()
        good_rate = beta_wf.sum()
        total_power_pgd += good_power
        total_power_baseline += good_power
        total_rate_pgd += good_rate
        total_rate_baseline += good_rate

    ### second stage: either refer to PGD or trivial solution
    else: 
        bad_chan_times += 1

        # feasi_flag, sum_rate_trivial, _, sum_power_trivial = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon)
        # if feasi_flag == True:
        #     trivial_feasible_times += 1
        #     total_power_trivial += sum_power_trivial
        #     total_rate_trivial += sum_rate_trivial
        
        ### trivial solver declare infeasibility
        if sum_rate_trivial_small < rate_require:
            baseline_infeasible_times += 1

        ### trivial solver declare feasibility
        else:
            baseline_feasi_times += 1
            feasi_flag, sum_rate_trivial, _, sum_power_trivial = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon)
            total_power_baseline += sum_power_trivial
            total_rate_baseline += sum_rate_trivial

            ### Third stage: PGD solver
            rate_opt_allo, _, _, _, power_pgd, rate_pgd, D_val_max = projected_gradient_descent_with_penalty(
                h_samples, g_samples, beta_wf,
                sigma_n, sigma_v,
                rate_require, epsilon,
                alpha_beta=1e-6, alpha_b=1e-6,
                gamma=0.2, C=20.0,
                max_iter=1000, tol=1e-6
            )

            if D_val_max < -1e-1:
                ### can only refer to the trivial solution
                # _, sum_rate_trivial, _, sum_power_trivial = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon)
                total_power_pgd += sum_power_trivial
                total_rate_pgd += sum_rate_trivial
            else: 
                ### adopt PGD converged results
                total_power_pgd += power_pgd
                total_rate_pgd += rate_pgd

# good_chan_times = 0
power_ave_pgd = total_power_pgd / (baseline_feasi_times+good_chan_times)
power_ave_baseline = total_power_baseline / (baseline_feasi_times+good_chan_times)
power_ave_trivial = total_power_trivial / trivial_feasible_times
rate_ave_pgd = total_rate_pgd / (baseline_feasi_times+good_chan_times)
rate_ave_baseline = total_rate_baseline / (baseline_feasi_times+good_chan_times)
rate_ave_trivial = total_rate_trivial / trivial_feasible_times

print(f"power_ave_pgd = {power_ave_pgd}, power_ave_baseline = {power_ave_baseline}, power_ave_trivial = {power_ave_trivial}")
print(f"rate_ave_pgd = {rate_ave_pgd}, rate_ave_baseline = {rate_ave_baseline}, rate_ave_trivial = {rate_ave_trivial}")
print(f"good_chan_times = {good_chan_times}, baseline_feasi_times = {baseline_feasi_times}")
bp()



for epoch in range(total_simu_times):
    print(f"Epoch {epoch}")
    h_samples = h_value_range[th.multinomial(prob_h, total_block, replacement=True)] 
    g_samples = g_value_range[th.multinomial(prob_g, total_block, replacement=True)]
    p_allo_up = epsilon / g_samples

    beta_wf, _ = solve_rate_allocation(h_samples, g_samples, rate_require, epsilon, sigma_n, tol=1e-9, max_iter=1000)
    feasi_flag, sum_rate_trivial, _, sum_power_trivial = baseline_nonconvex_solution(h_samples,g_samples,rate_require,sigma_n,sigma_v,epsilon)

    ### trivial feasibility test
    h_samples = h_samples.numpy()
    g_samples = g_samples.numpy()
    flag = infeasible_check(h_samples, g_samples, sigma_n, sigma_v, epsilon, rate_require)

    if flag:
        infeasible_times += 1
    # elif sum_rate_trivial >= rate_require:
    elif feasi_flag == True:
        trivial_feasible_times += 1
    else:
        continue

# feasi_prob_trivial_scheme = trivial_feasible_times / total_simu_times
# print(f"feasi_prob_trivial_scheme = {feasi_prob_trivial_scheme}")
# bp()

    h_samples = h_samples.numpy()
    g_samples = g_samples.numpy()

    flag = infeasible_check(h_samples, g_samples, sigma_n, sigma_v, epsilon, rate_require)
    gap = delta_func(beta_wf, h_samples, g_samples, sigma_n, sigma_v)

    ### not pass the infeasibility check
    if flag:
        infeasible_times += 1

    ### first stage: convex solution is feasible
    elif gap >= 0:
        good_chan_times += 1

    ### second stage: either refer to PGD or trivial solution
    else: 
        bad_chan_times += 1
        rate_opt_allo, _, _, _, power_pgd, _, D_val_max = projected_gradient_descent_with_penalty(
            h_samples, g_samples, beta_wf,
            sigma_n, sigma_v,
            rate_require, epsilon,
            alpha_beta=1e-6, alpha_b=1e-6,
            gamma=0.2, C=20.0,
            max_iter=600, tol=1e-6
        )

        ## exist infeasible cases even converged
        # if less_noisy_check(h_samples, g_samples, sigma_n, sigma_v, P_opt_allo) == False:
        if D_val_max < -1e-1:
            pgd_infeasible_times += 1
            if sum_rate_trivial >= rate_require:
                pgd_baseline_feasi_times += 1
        else:
            pgd_feasi_times += 1


        if sum_rate_trivial < rate_require:
            baseline_infeasible_times += 1
        else:
            baseline_feasi_times += 1

feasi_prob_pgd_scheme = (good_chan_times + pgd_feasi_times + pgd_baseline_feasi_times) / (total_simu_times - infeasible_times)
feasi_prob_convex_scheme = (good_chan_times + baseline_feasi_times) / (total_simu_times - infeasible_times)

# print(f"infeasible_times: {infeasible_times}, good_chan_times = {good_chan_times}, bad_chan_times = {bad_chan_times}, pgd_feasi_times = {pgd_feasi_times}, pgd_infeasible_times = {pgd_infeasible_times}, baseline_feasi_times = {baseline_feasi_times}, baseline_infeasible_times = {baseline_infeasible_times}")
# print(f"infeasi_check_pass_prop = {infeasi_check_pass_prop}, good_prop = {good_prop}, pgd_feasi_prop = {pgd_feasi_prop}, pgd_baseline_feasi_prop = {pgd_baseline_feasi_prop}")
# print(f"power_ave_pgd = {power_ave_pgd}, power_ave_baseline = {power_ave_baseline}")
print(f"feasi_prob_pgd_scheme = {feasi_prob_pgd_scheme}, feasi_prob_convex_scheme = {feasi_prob_convex_scheme}") ## , feasi_prob_trivial_scheme = {feasi_prob_trivial_scheme}
print(f"{good_chan_times}, {bad_chan_times}, {infeasible_times}")
bp()

