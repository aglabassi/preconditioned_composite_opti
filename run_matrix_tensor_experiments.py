"""
Created on Wed Feb 19 14:34:20 2025
Updated on Mon Apr 28 2025

@author: aglabassi
"""

import torch
import os 
import numpy as np
from gauss_newton_actions import GN_action_matrix_tensor
from utils import (LinearMeasurementOperator, generate_data_and_initialize,
                   compute_gradient, compute_stepsize_and_damping,
                   cg_solve, split, ad_hoc_matrix_sensing)
from methods import (
    subgradient_method,
    GN_subgradient_method,
    LM_subgradient_method
)

def run_matrix_tensor_sensing_experiments(methods_test, experiment_setups, n1, n2, n3, r_true,
                 m_divided_by_r, identity, device,
                 n_iter, base_dir, loss_ord, initial_relative_error,
                 symmetric, tensor=True, had=False, corr_level=0,
                 geom_decay=False, q=0.97, lambda_=1e-5,
                 gamma=1e-8, gamma_custom=None):

    outputs = {}
    for r, kappa in experiment_setups:
        m = m_divided_by_r * r
        Aop = LinearMeasurementOperator(n1, n2, n3, m, device,
                                        identity=identity,
                                        tensor=tensor)

        T_star, y_obs, factors0, sizes = generate_data_and_initialize(
            measurement_operator=Aop, n1=n1, r_true=r_true, r=r,
            n2=n2, n3=n3, device=device, kappa=kappa,
            corr_level=corr_level, symmetric=symmetric,
            tensor=tensor, initial_relative_error=initial_relative_error
        )
        if tensor:
            X0, Y0, Z0 = factors0
        else:
            X0, Y0 = factors0
            Z0 = None

        def pack(X, Y, Z=None):
            if symmetric:
                # only pack X; for tensor we’ll replicate it in unpack
                return X.reshape(-1)
            elif tensor:
                return torch.cat([X.reshape(-1),
                                  Y.reshape(-1),
                                  Z.reshape(-1)])
            else:
                return torch.cat([X.reshape(-1), Y.reshape(-1)])

        def unpack(x):
            if symmetric:
                # unpack only X; Y (and Z if tensor) are the same
                X = x.reshape(n1, r)
                Y = X
                Z = X if tensor else None
                return X, Y, Z

            # non-symmetric case
            offset = n1 * r
            X = x[:offset].reshape(n1, r)
            Y = x[offset:offset + n2 * r].reshape(n2, r)
            if tensor:
                Z = x[offset + n2 * r:].reshape(n3, r)
                return X, Y, Z
            else:
                return X, Y, None

        # residual → subgradient
        def make_subgradient_fc(y_obs):
            def subgradient_fc(x):
                X, Y, Z = unpack(x)
                T = (torch.einsum('ir,jr,kr->ijk', X, Y, Z)
                     if tensor else X @ Y.T)
                res = Aop.A(T) - y_obs
                if loss_ord == 1:
                    return Aop.A_adj(torch.sign(res)).view(-1)
                elif loss_ord == 0.5:
                    return Aop.A_adj(res / torch.norm(res)).view(-1)
                elif loss_ord == 2:
                    return Aop.A_adj(res).view(-1)
                else:
                    raise ValueError("unsupported loss_ord")
            return subgradient_fc

        # Jacobianᵀ action → gradient
        def action_nabla_F_transpose_fc(x, v):
            X, Y, Z = unpack(x)
            return compute_gradient(
                X, Y, Z, v, n1, n2, n3,
                symmetric=symmetric, tensor=tensor
            )
        
        def objective_valuation(x):
            X, Y, Z = unpack(x)
            T = (torch.einsum('ir,jr,kr->ijk', X, Y, Z)
                 if tensor else X @ Y.T)
            res = Aop.A(T) - y_obs
            if loss_ord == 1:
                h_c_x =  torch.sum(torch.abs(res))
            elif loss_ord == 0.5:
                h_c_x =  torch.sum(res**2)**0.5
            elif loss_ord == 2:
                h_c_x =  torch.sum(res**2)*0.5
            else:
                raise ValueError("unsupported loss_ord")
            return h_c_x
        # wrappers for stepsize & damping
        def make_stepsize_damping_fc(method):
            def stepsize_damping_fc(k, x, precond_grad=None):
                
               
                    
                subgrad = make_subgradient_fc(y_obs)(x)
                grad = action_nabla_F_transpose_fc(x, subgrad)
                h_c_x = objective_valuation(x)
                
                stepsize, damping = compute_stepsize_and_damping(
                    method, grad, subgrad, h_c_x, loss_ord,
                    symmetric, tensor=tensor,
                    geom_decay=geom_decay, q=q, lambda_=lambda_,
                    gamma=gamma, k=k, X=None, Y=None, G=None,
                    device=device, gamma_custom=gamma_custom)
                
                return stepsize, damping
            return stepsize_damping_fc
        
        
        def stepsize_fc_lm(k,x, precond_grad=None):#for noiseless problems. foir noisy use other function wirth geom decay
            after_gram = GN_action_matrix_tensor(precond_grad, unpack(x), symmetric, tensor=tensor)
            return gamma_custom*(objective_valuation(x) - 0)/(torch.sum(precond_grad * after_gram))
            
  

        def lm_solver(x, damping, b):
            sol = cg_solve(
                lambda Δ: GN_action_matrix_tensor(Δ, unpack(x),
                                                  symmetric, tensor=tensor),
                b, damping
            )
            return sol
        

        for method in methods_test:
            print(f"\n{'-'*40}\nMethod: {method}\n{'-'*40}")
            x0 = pack(X0, Y0, Z0)
            subgradient_fc = make_subgradient_fc(y_obs)
            stepsize_damping_fc   = make_stepsize_damping_fc(method)
            stepsize_fc = lambda k,x:  stepsize_damping_fc(k,x)[0]
            damping_fc  = lambda k,x:  stepsize_damping_fc(k,x)[1]
            reconstruct = True

            if method == 'Polyak Subgradient':
                xs, times = subgradient_method(
                    stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    x0,
                    n_iter
                )
            elif method == 'Gauss-Newton':
                xs, times = GN_subgradient_method(
                    stepsize_fc_lm if not geom_decay else stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    # GN uses zero damping
                    lambda x, g: lm_solver(x, 0.0, g),
                    x0,
                    n_iter
                )
            elif method == 'Levenberg-Marquardt (ours)':
                xs, times = LM_subgradient_method(
                    stepsize_fc_lm if not geom_decay else stepsize_fc,
                    damping_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    lm_solver,
                    x0,
                    n_iter, geom_decay=geom_decay
                )
            else:
                errs,times =  ad_hoc_matrix_sensing(
                        X0.clone(), Y0.clone(), Z0.clone() if Z0 is not None else Z0,
                        T_star, Aop, y_obs,
                        n_iter, method, loss_ord, m,
                        n1, n2, n3, sizes, split,
                        symmetric=symmetric,
                        tensor=tensor,
                        geom_decay=geom_decay, q=q,
                        lambda_=lambda_, gamma=gamma,
                        gamma_custom=gamma_custom,
                        device=device
                    )
                reconstruct = False
                                

            # now reconstruct errs from the trajectory xs
            if reconstruct:
                errs = []
                for k,x in enumerate(xs):
                    X, Y, Z = unpack(x)
                    T = (torch.einsum('ir,jr,kr->ijk', X, Y, Z)
                         if tensor else X @ Y.T)
                    rel_err = torch.norm(T - T_star) / torch.norm(T_star)
                    errs.append(rel_err.item())
                    if k % 20 == 0:
                        print(f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}")

            # save, print, etc.
            outputs[method] = errs
            fname = f'exp{"tensor" if tensor else "matrix"}{"sym" if symmetric else ""}_' \
                    f'{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={kappa}_trial_0.csv'
            fname_times = f'times{"tensor" if tensor else "matrix"}{"sym" if symmetric else ""}_' \
                    f'{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={kappa}_trial_0.csv'
            np.savetxt(os.path.join(base_dir, fname), np.array(errs), delimiter=',')
            np.savetxt(os.path.join(base_dir, fname_times), np.array(times), delimiter=',')

    return outputs
