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

##### schemes sumary #####
## baseline: feasibility check + water-filling + if not feasible, adopt trivial feasible solution
## proposed method: feasibility check + water-filling + if not feasible, PGA + if PGA not feasible, adopt trivial feasible solution

##### probability to be tested #####
## 1. the probability that the basic feasibility holds
## 2. the probability that convex solution is feasible given condition 1
## 3. the probability that PGA converges to a feasible solution given condition 2


cuda = True if th.cuda.is_available() else False
device = th.device("cuda" if cuda else "cpu")

scale_h = 1
scale_g = 1
sigma_n = 0.31
sigma_v = 0.31
epsilon = 1
quan_step = 0.01
power_step = 0.01


def feasible_check(h, g, sigma_n, sigma_v):
    flag = False
    L = len(h)
    for l in range(L):
        if h[l]/sigma_n >= g[l]/sigma_v:
            flag = True
            break
    return flag

def total_power_given_eta(eta, p_allo_up, h_samples):
    temp = np.maximum(0.0, eta - sigma_n/h_samples)
    temp = np.minimum(temp, p_allo_up)
    return np.sum(temp)

def solve_power_allocation(h, g, P0, epsilon, sigma_n, tol=1e-9, max_iter=1000):
    h = np.array(h, dtype=float)
    g = np.array(g, dtype=float)
    L = len(h)

    P_ub = epsilon / g  
    if np.sum(P_ub) <= P0:
        return P_ub
    
    ### else: P_l = min( epsilon/g_l , max(0, eta - sigma_n/h_l ) )
    left = 0.0
    right = np.max(sigma_n/h + P_ub) + 1.0

    ### binary search
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        current_sum = total_power_given_eta(mid, P_ub, h)

        if abs(current_sum - P0) < tol:
            break
        if current_sum > P0:
            right = mid
        else:
            left = mid

    ## obtain the optimal power allo based on converged eta
    eta_star = 0.5 * (left + right)
    P_opt = np.maximum(0.0, eta_star - sigma_n / h)
    P_opt = np.minimum(P_opt, P_ub)

    return P_opt


### less noisy constraint calculation
def delta_func(P, h, g, sigma_n, sigma_v):
    term1 = np.log1p(P * h / sigma_n).sum()   # log(1 + P_l*h_l/sigma_n)
    term2 = np.log1p(P * g / sigma_v).sum()  # log(1 + P_l*g_l/sigma_v)
    return term1 - term2

## \partial \Delta / \partial P_\ell
def grad_delta(P, h, g, sigma_n, sigma_v):
    grad1 = h / (sigma_n + P * h)     # h_l/(sigma_n + P_l*h_l)
    grad2 = g / (sigma_v + P * g)     # g_l/(sigma_v + P_l*g_l)
    return grad1 - grad2 ## vector

## \partial \mathcal{F} / \partial P_\ell
def grad_objective(P, h, g, sigma_n, sigma_v, eta, b):
    grad_orig = h / (sigma_n + P * h)   # d/dP_l of sum_{l} log(1+...)
    dDelta = grad_delta(P, h, g, sigma_n, sigma_v)
    Dval = delta_func(P, h, g, sigma_n, sigma_v)
    return grad_orig - 2.0 * eta * (Dval - b) * dDelta

## \partial \mathcal{F} / \partial b
def grad_b(P, h, g, sigma_n, sigma_v, eta, b):
    Dval = delta_func(P, h, g, sigma_n, sigma_v)
    return 2.0 * eta * (Dval - b)

## iteratively project P onto the feasible set
def project_onto_P(P, g, P0, epsilon, pocs_max_iter=100, pocs_tol=1e-9):
    P_proj = P.copy()
    for _ in range(pocs_max_iter):
        P_prev = P_proj.copy()

        ## project to P_l <= epsilon / g_l
        upper_bound = epsilon / g
        P_proj = np.minimum(P_proj, upper_bound)
        P_proj = np.maximum(P_proj, 0.0)

        ## project to sum_l P_l <= P0
        s = P_proj.sum()
        if s > P0:
            P_proj *= (P0 / s)

        # convergence check
        if np.linalg.norm(P_proj - P_prev) < pocs_tol:
            break

    return P_proj



###################################
# Parameter description

# P_init: the initial point (vector), which may violate the new constraint (i.e., Δ(P_init) < 0).
# C: a large constant for setting the initial penalty factor η^(0) as in Eq. (16) of the reference.
# gamma: controls the growth of η^(n) via 1 + gamma·n².
# alpha_P, alpha_b: step sizes for the gradient updates of P and b, respectively.
# tol: the stopping threshold δ, which requires both ‖P^(n+1) – P^(n)‖ and |b^(n+1) – b^(n)| to be sufficiently small, and also Δ(P) ≥ 0.
# Returns
# (P_opt, b_opt, history), where history stores iteration information for examining the convergence process.

def projected_gradient_descent_with_penalty(
        h, g,
        P_init,      ## water-filling solution
        sigma_n, sigma_v,
        P0, epsilon,
        alpha_P, alpha_b,   # step size of P and b updates
        gamma,                   # eta(n) = eta(0)*(1+gamma*n^2)
        C,                      # related to the initial eta(0)
        max_iter, tol
    ):

    P = P_init.copy()
    b = 0.0  
    L = len(P)

    D0 = delta_func(P, h, g, sigma_n, sigma_v)
    sum_log = np.log1p(P * h / sigma_n).sum()
    # if D0 < 0.0:
    #     eta0 = (C / (D0**2)) * max(sum_log, 1e-6)  
    # else:
    #     eta0 = 1e-6

    eta0 = (C / (D0**2)) * max(sum_log, 1e-6)

    history_rate = []
    history_delta = []
    history_b = []
    D_val_max = -1e8
    for n in range(max_iter):
        if n % 30 == 0:
        #eta_n = eta0 * (1.0 + gamma*(n**2))
            update_time = n // 30
            eta_n = eta0 * (1.0 + gamma*update_time)

        ## gradient descent of P
        gradient_P = grad_objective(P, h, g, sigma_n, sigma_v, eta_n, b)
        P_tmp = P + alpha_P * gradient_P

        ## project
        P_new = project_onto_P(P_tmp, g, P0, epsilon)

        ## gradient descent of b
        gb = grad_b(P_new, h, g, sigma_n, sigma_v, eta_n, b)
        b_new = b + alpha_b * gb
        b_new = max(b_new, 0.0)

        ## convergence conditions: dP, db, Dval
        dP = np.linalg.norm(P_new - P)
        db = abs(b_new - b)
        Dval = delta_func(P_new, h, g, sigma_n, sigma_v)
        D_val_max = max(D_val_max, Dval)
        
        # update
        P, b = P_new, b_new
        rate_new = np.log1p(P * h / sigma_n).sum()

        history_rate.append(rate_new)
        history_delta.append(Dval)
        history_b.append(b)

        # if n % 10 == 0:
        #     print(f"Iteration {n}: Rate = {rate_new}, Delta = {Dval}, dP = {dP}, db = {db}, Dval = {Dval}")

        ## stopping criteria
        if dP < tol and db < tol and Dval >= 0:  # Dval >= -1e-9 to account for numerical errors
            break
    rate_final = rate_new
    return P, b, history_rate, history_delta, history_b, rate_final, D_val_max

def baseline_nonconvex_solution(h,g,total_power,sigma_n,sigma_v,epsilon):
    L = len(h)
    P = np.zeros(L)
    bound_prop = 0
    for l in range(L):
        if h[l]/sigma_n >= g[l]/sigma_v:
            bound_prop += epsilon/g[l]
            P[l] = epsilon/g[l]
    if bound_prop > total_power:
        P = P * (total_power / bound_prop)
    sum_rate = np.log1p(P * h / sigma_n).sum()
    sum_power = np.sum(P)
    return P, sum_rate, sum_power

def less_noisy_check(h, g, sigma_n, sigma_v, P_allo):
    L = len(h)
    flag = False
    if np.log1p(P_allo * h / sigma_n).sum() + 1e-4 >= np.log1p(P_allo * g / sigma_v).sum():
        flag = True
    return flag



######### Monte-Carlo simulation #########
total_block = 10
total_power = 6
total_simu_times = 10000
infeasible_times = 0
good_chan_times = 0
bad_chan_times = 0
bad_chan_feasi_times = 0
pga_infeasible_times = 0
baseline_infeasi_times = 0
good_chan_sum_rate = 0
bad_chan_sum_rate_pga = 0
bad_chan_sum_rate_baseline = 0
sum_rate_pga = 0
sum_rate_baseline = 0
sum_power_pga = 0
sum_power_baseline = 0
trivial_sum_rate = 0
trivial_sum_power = 0

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


###### a single test #######

flag = False
while flag == False or gap >= 0:

    h_samples = h_value_range[th.multinomial(prob_h, total_block, replacement=True)] 
    g_samples = g_value_range[th.multinomial(prob_g, total_block, replacement=True)]
    th.save(h_samples, 'h_samples_power.pth')
    th.save(g_samples, 'g_samples_power.pth')

    P_wf = solve_power_allocation(h_samples, g_samples, total_power, epsilon, sigma_n, tol=1e-9, max_iter=1000)

    h_samples = h_samples.numpy()
    g_samples = g_samples.numpy()
    P_base, _, _ = baseline_nonconvex_solution(h_samples, g_samples, total_power, sigma_n, sigma_v, epsilon)

    flag = feasible_check(h_samples, g_samples, sigma_n, sigma_v)
    gap = delta_func(P_wf, h_samples, g_samples, sigma_n, sigma_v)

h_samples = th.load('h_samples_power.pth')
g_samples = th.load('g_samples_power.pth')
# h_samples = th.load('h_samples.pth')
# g_samples = th.load('g_samples.pth')
P_wf = solve_power_allocation(h_samples, g_samples, total_power, epsilon, sigma_n, tol=1e-9, max_iter=1000)
h_samples = h_samples.numpy()
g_samples = g_samples.numpy()
P_base, _, _ = baseline_nonconvex_solution(h_samples, g_samples, total_power, sigma_n, sigma_v, epsilon)

random_values = np.random.rand(total_block)
normalized_random_values = random_values / np.sum(random_values)
P_rand = total_power * normalized_random_values


# if not flag:
#     print("infeasible!")
# elif gap >= 0:
#     P_opt_allo = P_wf
# else:  ### currently "alpha = 1e-6, gamma = 0.2, C = 20.0" is the best


#### initialized with water-filling solution
P_opt_allo, b_opt, hist_rate, hist_delta, hist_b, sum_rate_pga, _ = projected_gradient_descent_with_penalty(
    h_samples, g_samples, P_wf,
    sigma_n, sigma_v,
    total_power, epsilon,
    alpha_P=1e-6, alpha_b=1e-6,
    gamma=gamma, C=C_init,
    max_iter=max_iter, tol=1e-6
)

print(f"sum_rate = {sum_rate_pga}")

iterations = range(len(hist_rate))


#### initialized with trivial allocation solution
P_opt_allo_2, b_opt_2, hist_rate_2, hist_delta_2, hist_b_2, sum_rate_pga_2, _ = projected_gradient_descent_with_penalty(
    h_samples, g_samples, P_base,
    sigma_n, sigma_v,
    total_power, epsilon,
    alpha_P=1e-6, alpha_b=1e-6,
    gamma=gamma, C=C_init,
    max_iter=max_iter, tol=1e-6
)


#### initialized with random allocation solution
P_opt_allo_3, b_opt_3, hist_rate_3, hist_delta_3, hist_b_3, sum_rate_pga_3, _ = projected_gradient_descent_with_penalty(
    h_samples, g_samples, P_rand,
    sigma_n, sigma_v,
    total_power, epsilon,
    alpha_P=1e-6, alpha_b=1e-6,
    gamma=gamma, C=C_init,
    max_iter=max_iter, tol=1e-6
)

# P_baseline_allo, sum_rate_baseline = baseline_nonconvex_solution(h_samples,g_samples,total_power,sigma_n,sigma_v,epsilon)
# print(f"sum_rate = {sum_rate_pga}, sum_rate_baseline = {sum_rate_baseline}")

# interval = 30
# plt.xticks(np.arange(0, len(hist_rate), interval), labels=[f'{i*10}' for i in np.arange(0, len(hist_rate), interval)])

def pad_sequence(seq, max_length):
    return np.pad(seq, (0, max_length - len(seq)), 'constant', constant_values=np.nan)

max_length = max(len(hist_rate), len(hist_rate_2), len(hist_rate_3))
hist_rate = pad_sequence(hist_rate, max_length)
hist_rate_2 = pad_sequence(hist_rate_2, max_length)
hist_rate_3 = pad_sequence(hist_rate_3, max_length)

max_length_delta = max(len(hist_delta), len(hist_delta_2), len(hist_delta_3))
hist_delta = pad_sequence(hist_delta, max_length_delta)
hist_delta_2 = pad_sequence(hist_delta_2, max_length_delta)
hist_delta_3 = pad_sequence(hist_delta_3, max_length_delta)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(iterations, hist_rate, marker='o', linestyle='-', color='b', label='Rate, convex initialization')
# plt.plot(iterations, hist_b, marker='*', linestyle='-', color='r', label='auxiliary b, convex initialization')
plt.plot(iterations, hist_rate_2, marker='*', linestyle='-', color='r', label='Rate, trivial initialization')
plt.plot(iterations, hist_rate_3, marker='^', linestyle='-', color='g', label='Rate, random initialization')
plt.xlabel('Iteration n')
# plt.xlabel('Update step k')
plt.ylabel('Sum covert rate')
plt.title('Sum covert rate')
plt.grid(True, which="both", ls="--")
plt.legend()

interval = 600
plt.xticks(np.arange(0, len(iterations), interval), labels=[f'{i//30}' for i in np.arange(0, len(iterations), interval)])

plt.subplot(1, 2, 2)
plt.plot(iterations, hist_delta, marker='o', linestyle='-', color='b', label='Delta, convex initialization')
plt.plot(iterations, hist_delta_2, marker='*', linestyle='-', color='r', label='Delta, trivial initialization')
plt.plot(iterations, hist_delta_3, marker='^', linestyle='-', color='g', label='Delta, random initialization')
plt.xlabel('Iteration n')
# plt.xlabel('Update step k')
# plt.ylabel('Delta function')
plt.title('Feasibility')
plt.grid(True, which="both", ls="--")
plt.legend()

interval = 600
plt.xticks(np.arange(0, len(iterations), interval), labels=[f'{i//30}' for i in np.arange(0, len(iterations), interval)])

plt.tight_layout()
plt.show()

# print(f"optimal power allocation: {P_opt_allo}, sum = {np.sum(P_opt_allo)}")

# plt.figure(figsize=(10, 6))
# plt.bar(range(len(P_opt_allo)), P_opt_allo, color='blue')
# plt.xlabel('Index')
# plt.ylabel('Power Level')
# plt.title('Power Allocation')
# plt.show()
