#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:17:57 2025

@author: aglabassi
"""

import torch
import os 
import numpy as np
from methods import subgradient_method, LM_subgradient_method, GN_subgradient_method


def generate_point_on_boundary_positive_orthant(p, epsilon):

    radius = torch.norm(p) * epsilon

    
    random_direction = torch.abs(torch.randn_like(p))
    random_direction = random_direction / torch.norm(random_direction)

    new_point = p + radius * random_direction

    return new_point

def generate_difficult_A(m, n, kappa_A=10, device='cpu'):
    
    s_values = torch.linspace(1, 1 / kappa_A, n, device=device)

    U, _ = torch.linalg.qr(torch.randn(m, m, device=device))
    V, _ = torch.linalg.qr(torch.randn(n, n, device=device))

    Sigma = torch.zeros((m, n), device=device)
    for i in range(min(m, n)):
        Sigma[i, i] = s_values[i]

    A = U @ Sigma @ V.T
    return A

def setup_experiment(m, n, r_true, kappa, initial_rel_err, device='cpu'):
    """
    Set up the experiment using PyTorch.
    
    Parameters:
      m : int, number of rows of A
      n : int, number of columns of A
      r_true : int, rank for true x_star
      kappa : float, condition number
      device : str, device ('cpu' or 'cuda')
    
    Returns:
      x_star, x0, T, A, b, methods, base_dir
    """

    # Create x_star
    x_star = torch.cat((
        torch.linspace(1, 1/kappa, r_true, device=device),
        torch.zeros(n - r_true, device=device)
    ))


    z0 = generate_point_on_boundary_positive_orthant(x_star**2, initial_rel_err)
    x0 = torch.sqrt(z0)


    A = generate_difficult_A(m, n,device=device)

    b = A @ (x_star**2)


    return x_star, x0, A, b



def run_nonegative_least_squares_experiments(methods_test, experiment_setups, r_true,
                 m_divided_by_r, device,
                 n_iter, base_dir, initial_rel_err, loss_ord):
    
    outputs = {}
    for r, kappa in experiment_setups:
        
        m = m_divided_by_r*r
        n = r
        
        x_star, x0, A, b = setup_experiment(m,n,r_true,kappa, initial_rel_err, device=device)
        
        for method in methods_test:
            
            def stepsize_fc(k,x, precond_grad=None):
                
                if loss_ord == 2:
                    numerator  =  0.5*( torch.linalg.norm( A@(x*x) - b , ord=2)**2) - 0
                elif loss_ord == 1:
                    numerator = torch.linalg.norm(A@(x*x) - b, ord=1)
                elif loss_ord == 0.5:
                    numerator  =  torch.linalg.norm( A@(x*x) - b , ord=2)
                if method not in ['Polyak Subgradient']:     
                    denominator = torch.sum(precond_grad * action_nabla_F_transpose_fc(x, action_nabla_F_transpose_fc(x, precond_grad )))
                else:
                    grad = action_nabla_F_transpose_fc(x, subgradient_fc(x))
                    denominator = torch.sum(grad*grad)
                return 0.5*numerator/denominator
                
            def damping_fc(k,x):
                if loss_ord == 2:
                    return 10**-2 *torch.sqrt( ( torch.linalg.norm( A@(x*x) - b , ord=2)**2))
                elif loss_ord==0.5:
                    return 10**-2* torch.linalg.norm( A@(x*x) - b , ord=2)
                elif loss_ord == 1:
                    return 10**-2* torch.linalg.norm( A@(x*x) - b , ord=1) 
            def subgradient_fc(x):
                if loss_ord ==2:
                    res = A.T@(A@(x*x) - b )
                elif loss_ord ==1:
                    res =  A.T@(torch.sign(A@(x*x) - b ))
                elif loss_ord==0.5:
                    res = A.T@(A@(x*x) - b ) / (torch.linalg.norm( A@(x*x) - b))
                return res 
            def action_nabla_F_transpose_fc(x,v):
                return 2*(x*v)
            
            def lm_solver(x,damping, b):
                return b*(1/(4*(x*x)+damping))

            if method == 'Polyak Subgradient':
                xs,_ = subgradient_method(
                    stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    x0,
                    n_iter
                )
            
            elif method == 'Gauss-Newton':
                xs,_ = GN_subgradient_method(
                    stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    # GN uses zero damping
                    lambda x, g: lm_solver(x, 0.0, g),
                    x0,
                    n_iter
                )
            elif method == 'Levenberg-Marquardt (ours)':
                xs,_ = LM_subgradient_method(
                    stepsize_fc,
                    damping_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    lm_solver,
                    x0,
                    n_iter
                )
            else:
                raise Exception()
            
            
            
            
            errs = []
            for k,x in enumerate(xs):
                x = torch.tensor(x)
                rel_err = torch.norm(x**2 - x_star**2) / torch.norm(x_star**2)
                errs.append(rel_err.item())
                if k % 20 == 0:
                    print(f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}")

            outputs[method] = errs
            fname = f'exphad_{method}_l_{loss_ord}_r*={r_true}_r={n}_condn={kappa}_trial_{0}.csv'
            np.savetxt(os.path.join(base_dir, fname), np.array(errs), delimiter=',')
        
               
    return outputs
           


def run_linear_least_squares_experiments(
    methods_test,
    experiment_setups,
    r_true,
    m_divided_by_r,
    device,
    n_iter,
    base_dir,
    initial_rel_err,
    loss_ord,
):
   



    outputs = {}
    for r, kappa in experiment_setups:
        m = m_divided_by_r * r
        n = r

        # ------------------------------------------------------------------
        # Synthetic instance (same spectrum logic as before, but *linear*)
        # ------------------------------------------------------------------
        x_star = torch.cat(
            (
                torch.linspace(1.0, 1.0 / kappa, r_true, device=device),
                torch.zeros(n - r_true, device=device),
            )
        )

        # Initial point: small relative perturbation inside the positive orthant
        x0 = (
            generate_point_on_boundary_positive_orthant(x_star, initial_rel_err)
            .clone()
            .detach()
        )

        A = generate_difficult_A(m, n, device=device)
        b = A @ x_star

        # For closures inside every algorithm ------------------------------
        def residual(x):
            return A @ x - b

        # Polyak numerator --------------------------------------------------
        def _polyak_numerator(x):
            if loss_ord == 2:
                return 0.5 * torch.linalg.norm(residual(x), ord=2).pow(2)
            elif loss_ord == 1:
                return torch.linalg.norm(residual(x), ord=1)
            elif loss_ord == 0.5:
                return torch.linalg.norm(residual(x), ord=2)
            else:
                raise ValueError("loss_ord must be 2, 1 or 0.5")

        # Common oracles ----------------------------------------------------
        def subgradient_fc(x):
            r = residual(x)
            if loss_ord == 2:
                return A.T @ r
            elif loss_ord == 1:
                return A.T @ torch.sign(r)
            else:  # 0.5
                return (A.T @ r) / torch.linalg.norm(r, ord=2)

        def action_nabla_F_transpose_fc(x, v):  # J_F(x)^T v with F = Id is v.
            return v

        # Diagonal LM/GN solver -------------------------------------------
        def lm_solver(x, damping, g):
            AtA = A.T @ A
            n_local = AtA.shape[0]
            return torch.linalg.solve(
                AtA + damping * torch.eye(n_local, device=device), g
            )

        # Polyak stepsize ---------------------------------------------------
        def stepsize_fc(k, x, precond_grad=None):
            numerator = _polyak_numerator(x)
            if method not in [
                "Polyak Subgradient",
            ]:
                # Pre‑conditioned gradient supplied by outer algorithm
                denominator = torch.sum(
                    precond_grad
                    * action_nabla_F_transpose_fc(
                        x, action_nabla_F_transpose_fc(x, precond_grad)
                    )
                )
            else:
                g = action_nabla_F_transpose_fc(x, subgradient_fc(x))
                denominator = torch.sum(g * g)
            return 0.5 * numerator / denominator

        def damping_fc(k, x):
            if loss_ord == 2:
                return 1e-2 * torch.linalg.norm(residual(x), ord=2)
            elif loss_ord == 1:
                return 1e-2 * torch.linalg.norm(residual(x), ord=1)
            else:  # 0.5
                return 1e-2 * torch.linalg.norm(residual(x), ord=2)

        # ------------------------------------------------------------------
        # Loop through methods --------------------------------------------
        # ------------------------------------------------------------------
        for method in methods_test:
            if method == "Polyak Subgradient":
                xs, _ = subgradient_method(
                    stepsize_fc, subgradient_fc, action_nabla_F_transpose_fc, x0, n_iter
                )
            elif method == "Gauss-Newton":
                xs, _ = GN_subgradient_method(
                    stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    lambda x, g: lm_solver(x, 0.0, g),
                    x0,
                    n_iter,
                )
            elif method == "Levenberg-Marquardt (ours)":
                xs, _ = LM_subgradient_method(
                    stepsize_fc,
                    damping_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    lm_solver,
                    x0,
                    n_iter,
                )
            else:
                raise ValueError(f"Unknown method '{method}'.")

            # --------------------------------------------------------------
            # Compute & log relative errors -------------------------------
            # --------------------------------------------------------------
            errs = []
            for k, x in enumerate(xs):
                x = torch.as_tensor(x, device=device)
                rel_err = torch.linalg.norm(x - x_star) / torch.linalg.norm(x_star)
                errs.append(rel_err.item())
                if k % 20 == 0:
                    print(
                        f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}"
                    )

            outputs[method] = errs
            fname = (
                f"expls_{method}_l_{loss_ord}_r*={r_true}_r={n}_condn={kappa}_trial_0.csv"
            )
            np.savetxt(os.path.join(base_dir, fname), np.array(errs), delimiter=",")

    return outputs
