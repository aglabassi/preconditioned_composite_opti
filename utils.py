#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:25:33 2025

@author: aglabassi
"""
import torch
import torch.nn.functional as F



###############################################################################
# Measurement Operator Class
###############################################################################
class LinearMeasurementOperator:
    def __init__(self, n1, n2, n3, m,  device, identity=False, tensor=True, distribution='Gaussian'):
        """
        For CP–tensor models (tensor=True) the underlying measurement tensor
        is of shape (m, n1, n2, n3). For matrix factorization (tensor=False), it is
        of shape (m, n1, n2) (with n2=n1 in the symmetric case).
        """
        self.identity = identity
        self.m = m
        self.tensor = tensor
        if tensor:
            self.n1 = n1; self.n2 = n2; self.n3 = n3
            shape = (m, n1, n2, n3)
        else:
            self.n1 = n1; self.n2 = n2  # for symmetric, n2 equals n1.
            shape = (m, n1, n2)
        if not identity:
            if distribution == 'Gaussian':
                self.A_tensors = torch.randn(*shape, device=device) / torch.sqrt(torch.tensor(m, device=device, dtype=torch.float64))
            elif distribution == 'Bernoulli:':
                self.A_tensors = (torch.rand(*shape, device=device) < 0.5).to(torch.float64)

    def A(self, X):
        if self.identity:
            return X.flatten()
        else:
            # For CP–tensor: X has 3 indices; for matrix factorization: 2 indices.
            if self.tensor:
                return torch.einsum('ijkl,jkl->i', self.A_tensors, X)
            else:
                return torch.einsum('ijk,jk->i', self.A_tensors, X)

    def A_adj(self, y):
        if self.identity:
            # Identity operator: simply reshape.
            return y
        else:
            if self.tensor:
                y_expanded = y.reshape(self.m, 1, 1, 1)
                return torch.sum(y_expanded * self.A_tensors, dim=0)
            else:
                y_expanded = y.reshape(self.m, 1, 1)
                return torch.sum(y_expanded * self.A_tensors, dim=0)



def local_init(T_star, factors, dims, tol, r, symmetric, tensor=True):
    """
    Performs local initialization for both CP–tensor and matrix factorization.
    
    Args:
        T_star (torch.Tensor): The target tensor (or matrix).
        factors (list[torch.Tensor]): 
            - If tensor==True:
                • symmetric: [X_star] 
                • asymmetric: [X_star, Y_star, Z_star]
            - If tensor==False:
                • symmetric: [X_star]
                • asymmetric: [X_star, Y_star]
        dims (list[int]): The row–dimensions of each factor.
        tol (float): Tolerance for the relative error.
        r (int): Target number of columns (rank) after padding.
        symmetric (bool): Whether the model is symmetric.
        tensor (bool): If True, use CP–tensor reconstruction; else use matrix factorization.
    
    Returns:
        list[torch.Tensor]: The updated factor matrices.
    """
    
    if tol is None:

        if symmetric:
            f = (dims[0]**-0.5)*torch.randn(dims[0],r, device=T_star.device)
            return f,f,f
        else:
            f1 = (dims[0]**-0.5)*torch.randn(dims[0],r, device=T_star.device)
            f2 = (dims[1]**-0.5)*torch.randn(dims[1],r,device = T_star.device)
            f3 = (dims[2]**-0.5)*torch.randn(dims[2],r, device = T_star.device)
            return f1,f2,f3

    new_factors = []
    for fac in factors:
        pad_amount = r - fac.shape[1]
        new_factors.append(F.pad(fac, (0, pad_amount), mode='constant', value=0))
    
    if tensor:
        if symmetric:
            T = torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[0], new_factors[0])
        else:
            T =torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[1], new_factors[2])
    else:
        if symmetric:
            T = new_factors[0] @ new_factors[0].T
        else:
            T = new_factors[0] @ new_factors[1].T
    
    err_rel = torch.norm(T - T_star) / torch.norm(T_star)
    to_add = 1e-6
    while err_rel <= tol:
        
        if tensor:
            if symmetric:
                new_factors[0] = new_factors[0] + torch.rand(dims[0], r, device=new_factors[0].device) * to_add
            else:
                new_factors[0] = new_factors[0] + torch.randn(dims[0], r, device=new_factors[0].device) * to_add
                new_factors[1] = new_factors[1] + torch.randn(dims[1], r, device=new_factors[1].device) * to_add
                new_factors[2] = new_factors[2] + torch.randn(dims[2], r, device=new_factors[2].device) * to_add
        else:
            if symmetric:
                new_factors[0] = new_factors[0] + torch.rand(dims[0], r, device=new_factors[0].device) * to_add
            else:
                new_factors[0] = new_factors[0] + torch.randn(dims[0], r, device=new_factors[0].device) * to_add
                new_factors[1] = new_factors[1] + torch.randn(dims[1], r, device=new_factors[1].device) * to_add
        
        if tensor:
            if symmetric:
                T = torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[0], new_factors[0])
            else:
                T = torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[1], new_factors[2])
        else:
            if symmetric:
                T = new_factors[0] @ new_factors[0].T
            else:
                T = new_factors[0] @ new_factors[1].T
        
        err_rel = torch.norm(T - T_star) / torch.norm(T_star)
    return new_factors


def generate_data_and_initialize(
    measurement_operator,
    n1, 
    r_true, 
    r,
    n2=None, 
    n3=None,
    device='cpu',
    kappa=10.0,
    corr_level=0.0,
    symmetric=False,
    tensor=False,
    initial_relative_error=1e-1
):
    """
    Generates synthetic data (matrix or tensor) along with noise,
    then performs a local initialization for factorization.

    Parameters
    ----------
    measurement_operator : object
        An operator that has a method A(...) that applies the measurement.
    n1 : int
        Dimension 1 for the ground-truth factor(s).
    r_true : int
        True rank of the ground-truth factor(s).
    r : int
        Desired rank for initialization.
    n2 : int, optional
        Dimension 2 (required for asymmetric matrix or 3D tensor if not symmetric).
    n3 : int, optional
        Dimension 3 (required for 3D tensor).
    device : str, optional
        Torch device (e.g., 'cpu' or 'cuda').
    kappa : float, optional
        Ratio controlling the range of singular values (1.0 to 1/kappa).
    corr_level : float, optional
        If > 0, fraction of measurements to corrupt with noise.
    symmetric : bool, optional
        Whether the factorization is symmetric.
    tensor : bool, optional
        Whether to construct a 3D tensor or a matrix.
    initial_relative_error : float, optional
        Relative error level for local_init(...).

    Returns
    -------
    T_star (for metric evaluation purposes)
    y_observed : torch.Tensor
        The measurement vector (possibly corrupted by noise).
    factors : tuple
        The initialized factor(s). For a 3D tensor, returns (X0, Y0, Z0).
        For a matrix, returns (X0, Y0)
    sizes : list of torch.Size
        Shapes of the returned factor(s).
    """

    # -- 1) Construct ground‐truth factors.
    # Orthonormal bases for X
    ux, _ = torch.linalg.qr(torch.rand(n1, r_true, device=device))
    vx, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))
    singular_values = torch.linspace(1.0, 1.0 / kappa, r_true, device=device)
    S = torch.diag(singular_values)
    X_star = ux @ (S**( 0.5 if not tensor else 1/3 ))  # shape: (n1, r_true)

    # We'll create either a tensor T_star (3D) or a matrix T_star (2D) depending on flags
    if tensor:
        # 3D tensor
        if symmetric:
            # Symmetric => same factor X_star used in each mode
            T_star = torch.einsum('ir,jr,kr->ijk', X_star, X_star, X_star)
        else:
            # Asymmetric => create distinct Y_star, Z_star
            if n2 is None or n3 is None:
                raise ValueError("n2 and n3 must be provided for a 3D asymmetric tensor.")
            uy, _ = torch.linalg.qr(torch.rand(n2, r_true, device=device))
            uz, _ = torch.linalg.qr(torch.rand(n3, r_true, device=device))

            Y_star = uy @ (S**1/3) 
            Z_star = uz @ (S**1/3)
            T_star = torch.einsum('ir,jr,kr->ijk', X_star, Y_star, Z_star)
    else:
        # Matrix
        if symmetric:
            # Symmetric => T_star = X_star X_star^T
            T_star = X_star @ X_star.T
        else:
            # Asymmetric => create second factor Y_star
            if n2 is None:
                raise ValueError("n2 must be provided for asymmetric matrix factorization.")
            uy, _ = torch.linalg.qr(torch.rand(n2, r_true, device=device))
            Y_star = uy @ (S**0.5)
            T_star = X_star @ Y_star.T

    # Ensure T_star is on the correct device (usually already is, 
    # but this is a safety net):
    T_star = T_star.to(device)

    # -- 2) Compute measurements and possibly add noise.
    y_true = measurement_operator.A(T_star).to(device)

    # Generate y_false on the same device
    if not tensor:
        # For matrix case
        # Provide n2 if not symmetric
        shape_for_rand = (n1, n2) if n2 is not None else (n1, n1)
        y_false = measurement_operator.A(torch.rand(shape_for_rand, device=device))
    else:
        # For tensor case
        shape_for_rand = (n1, n2, n3)
        y_false = measurement_operator.A(torch.rand(shape_for_rand, device=device))

    num_ones = int(y_true.shape[0] * corr_level)
    perm = torch.randperm(y_true.shape[0], device=device)
    mask_indices = perm[:num_ones]

    mask = torch.zeros(y_true.shape[0], device=device)
    mask[mask_indices] = 1
    
    y_observed = (1 - mask) * y_true + mask * y_false

    # -- 3) Unified local initialization.
    # Note: we assume you have a function local_init(...) defined elsewhere.
    if tensor:
        if symmetric:
            new_factors = local_init(
                T_star, [X_star], [n1],
                initial_relative_error, r, True, tensor=True
            )
            X0 = Y0 = Z0 = new_factors[0]
            sizes = [X0.shape]
            factors = (X0, X0, X0)
        else:
            new_factors = local_init(
                T_star, [X_star, Y_star, Z_star], [n1, n2, n3],
                initial_relative_error, r, False, tensor=True
            )
            X0, Y0, Z0 = new_factors
            sizes = [X0.shape, Y0.shape, Z0.shape]
            factors = (X0, Y0, Z0)
    else:
        if symmetric:
            new_factors = local_init(
                T_star, [X_star], [n1],
                initial_relative_error, r, True, tensor=False
            )
            X0 = new_factors[0]
            sizes = [X0.shape]
            factors = (X0, X0)
        else:
            new_factors = local_init(
                T_star, [X_star, Y_star], [n1, n2],
                initial_relative_error, r, False, tensor=False
            )
            X0, Y0 = new_factors
            sizes = [X0.shape, Y0.shape]
            factors = (X0, Y0)

    return T_star, y_observed, factors, sizes




###############################################################################
# 3. Helper: Split a Flattened Tensor into Blocks
###############################################################################
def split(concatenated, shapes):
    """
    Splits a flattened tensor into blocks with specified shapes.
    
    Args:
        concatenated (torch.Tensor): A 1D tensor.
        shapes (list[tuple]): List of shapes, e.g. [(m, r), (n, r), ...].
    
    Returns:
        tuple[torch.Tensor]: The split tensors.
    """
    output = []
    start = 0
    for shape in shapes:
        numel = torch.prod(torch.tensor(shape))
        block = concatenated[start:start + numel].reshape(shape)
        output.append(block)
        start += numel
    return tuple(output)


###############################################################################
# Generic Conjugate–Gradient Solver
###############################################################################
def cg_solve(operator_fn, b, damping, max_iter=100, epsilon=1e-25):
    """
    Generic conjugate gradient solver.
    
    Solves operator(x) + damping*x = b for x, operator is a linear map.
    
    Args:
        operator_fn (callable): A function mapping x to A(x) (same shape as x).
        b (torch.Tensor): Right-hand side.
        damping (float): Damping parameter.
        max_iter (int): Maximum iterations.
        epsilon (float): Tolerance on the residual norm.
    
    Returns:
        torch.Tensor: The solution vector x.
    """
    x = torch.zeros_like(b)
    r = b - (operator_fn(x) + damping * x)
    p = r.clone()
    rs_old = (r * r).sum()
    for i in range(max_iter):
        Ap = operator_fn(p) + damping * p
        alpha = rs_old / (p * Ap).sum()
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new <= epsilon**2:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def  nabla_F_transpose_g(factors, v, symmetric, tensor=True):
    """
    Computes (nabla c)^T * v for both CP–tensor and matrix factorization.
    
    For CP–tensor (tensor=True):
      - Symmetric: returns a tensor of shape (n, r) computed via three Einstein sums.
      - Asymmetric: returns a tuple (A, B, C).
    
    For matrix factorization (tensor=False):
      - Symmetric: returns ((v + v^T) @ X).reshape(-1)
      - Asymmetric: returns ( (v @ Y).reshape(-1), (v^T @ X).reshape(-1) )
    
    Args:
        factors (list[torch.Tensor]): See description.
        v (torch.Tensor): 
            For tensor==True: 
                • symmetric: shape (n, n, n)
                • asymmetric: shape (n1, n2, n3)
            For tensor==False:
                • symmetric: either a flattened or (n, n) tensor.
                • asymmetric: either flattened or of shape (n1, n2).
        symmetric (bool): Whether the model is symmetric.
        tensor (bool): If True, CP–tensor formulas are used; else matrix factorization.
    
    Returns:
        For symmetric: torch.Tensor of shape (n, r) or flattened.
        For asymmetric: tuple of tensors.
    """
    if tensor:
        X, Y, Z = factors
        A = torch.einsum('ijk,jl,kl->il', v, Y, Z)
        B = torch.einsum('ijk,il,kl->jl', v, X, Z)
        C = torch.einsum('ijk,il,jl->kl', v, X, Y)
        return A + B + C if symmetric else (A, B, C) 
    else:
        if symmetric:
            X = factors[0]
            if v.dim() == 1:
                v_mat = v.reshape(X.shape[0], X.shape[0])
            else:
                v_mat = v
            return ((v_mat + v_mat.T) @ X)
        else:
            X, Y = factors
            if v.dim() == 1:
                v_mat = v.reshape(X.shape[0], Y.shape[0])
            else:
                v_mat = v
            g_x = (v_mat @ Y)
            g_y = (v_mat.T @ X)
            return (g_x, g_y)

def compute_gradient(
    X, 
    Y, 
    Z, 
    subgradient_h, 
    n1, 
    n2, 
    n3, 
    symmetric=False, 
    tensor=False
):
    """
    Computes the gradient given:
      - Factor(s) X, Y, Z
      - A subgradient `subgradient_h`
      - A function nabla_F_transpose_g(...) that does the adjoint operation
      - The dimensions n1, n2, n3
      - Flags indicating if the problem is symmetric and/or tensor-based.

    Returns
    -------
    grad : torch.Tensor
        A 1D tensor containing the concatenated gradient.
    """

    if tensor:
        # 3D tensor
        if symmetric:
            # subgradient_h is shaped [n1, n1, n1] => ensure correct view
            grad = nabla_F_transpose_g([X, X, X],
                                       subgradient_h.view(n1, n1, n1),
                                       True, 
                                       tensor=True).reshape(-1)
        else:
            # subgradient_h is shaped [n1, n2, n3]
            gX, gY, gZ = nabla_F_transpose_g([X, Y, Z],
                                             subgradient_h.view(n1, n2, n3),
                                             False, 
                                             tensor=True)
            grad = torch.cat((gX.reshape(-1), gY.reshape(-1), gZ.reshape(-1)))
    else:
        # Matrix
        if symmetric:
            # subgradient_h is shaped [n1, n1]
            grad = nabla_F_transpose_g([X, X],
                                       subgradient_h, 
                                       True, 
                                       tensor=False).reshape(-1)
        else:
            # subgradient_h is shaped [n1, n2]
            gX, gY = nabla_F_transpose_g([X, Y],
                                         subgradient_h, 
                                         False, 
                                         tensor=False)
            grad = torch.cat((gX.reshape(-1), gY.reshape(-1)))
            

    return grad



def compute_stepsize_and_damping(
    method,
    grad,
    subgradient_h,
    h_c_x,
    loss_ord,
    symmetric,
    tensor=False,
    geom_decay=False,
    lambda_=None,
    q=None,
    gamma=None,
    k=None,
    X=None,Y=None, G=None, device=None,gamma_one=False, gamma_custom=None,precond_grad=None,after_gram_precond_grad=None
):
    """
    Computes the stepsize and damping for various methods.

    Parameters
    ----------
    method : str
        The update method (e.g., 'Gradient descent', 'Scaled gradient', etc.).
    grad : torch.Tensor
        The gradient (flattened or otherwise). Used for 'Gradient descent'/'Polyak Subgradient'.
    subgradient_h : torch.Tensor
        The subgradient (same shape as needed in dot-product). Used in certain methods.
    h_c_x : float
        The scalar h(c(X, Y)) or a related objective value.
    loss_ord : int, optional
        The order of the loss function (only used for 'Levenberg-Marquardt (ours)' logic).
    geom_decay : bool, optional
        Flag to indicate geometric damping decay (for 'Levenberg-Marquardt (ours)' only).
    lambda_ : float, optional
        Base damping parameter if geom_decay is True.
    q : float, optional
        Decay rate if geom_decay is True.
    k : int, optional

    Returns
    -------
    stepsize : float
        The computed stepsize to be used for the update.
    damping : float
        The damping parameter used in some preconditioned methods.
    """

    # Default damping if not used in the method
    damping = 0.0
    constant_stepsize = 0.5

    if method in ['Gradient descent', 'Polyak Subgradient']:
        stepsize = h_c_x / (torch.norm(grad) ** 2)
        if geom_decay:
            stepsize = (q**k)*gamma
            if tensor and not symmetric:
                stepsize*=10

    elif method == 'OPSA($\lambda=10^{-8}$)':
        damping = 1e-8
        Gx,Gy = G
        aux_x = Gx@matrix_inverse_sqrt(Y.T@Y + damping*torch.eye(Y.shape[1],device=device), device)
        aux_y = Gy@matrix_inverse_sqrt(X.T@X + damping*torch.eye(X.shape[1], device=device), device)
        stepsize =  h_c_x / (torch.sum(aux_x**2) + torch.sum(aux_y**2) )
    
    elif method == 'Scaled gradient($\lambda=10^{-8}$)':
        damping = 1e-8
        stepsize = constant_stepsize 
    
    elif method == 'Precond. gradient':
        # Example: damping depends on sqrt(h_c_x)
        damping = torch.sqrt(torch.tensor(h_c_x)) * 2.5e-3
        stepsize = constant_stepsize
        
    elif method  in ['Levenberg-Marquardt (ours)', 'Gauss-Newton']:
        # Damping depends on loss_ord, plus possibly geometric decay
        if geom_decay:
            # damping = lambda_ * (q ** k)
            # Ensure lambda_, q, k are not None
            if lambda_ is None or q is None or k is None:
                raise ValueError("lambda_, q, k must be provided if geom_decay=True.")

            damping = lambda_ * (q**k)
            stepsize= gamma * (q**k)
        else:
            # fallback if not geometric
            if loss_ord == 2:
                damping = torch.sqrt(torch.tensor(h_c_x)) * 2.5e-3
            else:
                damping = h_c_x * 1e-3
                
            if symmetric and not tensor:
                gamma = 10 if loss_ord == 1 else 15
            elif not symmetric and not tensor:
                gamma = 5
            else:
                gamma=1
            if gamma_one:
                gamma=1
            if gamma_custom is not None:
                gamma=gamma_custom
            stepsize = (gamma*h_c_x) / torch.sum(subgradient_h*subgradient_h)

    else:
        raise NotImplementedError(f"Unknown method: {method}")
    return stepsize, (damping if method != 'Gauss-Newton' else 0)


def update_factors(
    X, 
    Y, 
    Z, 
    preconditioned_grad, 
    stepsize, 
    sizes, 
    split_fn,
    symmetric=False, 
    tensor=False
):
    """
    Updates the factors (X, Y, Z) in-place or via reassignment given a preconditioned gradient.
    
    Parameters
    ----------
    X : torch.Tensor
        Current factor for mode-1 (or the single factor if symmetric).
    Y : torch.Tensor or None
        Factor for mode-2 (or same as X if symmetric).
    Z : torch.Tensor or None
        Factor for mode-3 (used if tensor=True and not symmetric).
    preconditioned_grad : torch.Tensor
        The concatenated gradient, or the direct gradient (if symmetric).
    stepsize : float
        The scalar stepsize.
    sizes : list of torch.Size
        A list describing the shapes of each factor (e.g., [X.shape, Y.shape, Z.shape]).
    split_fn : callable
        A function that splits the flattened gradient into separate factor gradients.
        For instance: prgx, prgy, prgz = split_fn(preconditioned_grad, sizes).
    symmetric : bool
        If True, we assume a symmetric problem (X=Y=Z for a tensor).
    tensor : bool
        If True, indicates a 3D tensor problem; otherwise a 2D matrix problem.

    Returns
    -------
    X, Y, Z : torch.Tensor
        The updated factors. If symmetric, Y and Z will point to X.
    """

    if tensor:
        # 3D Tensor
        if symmetric:
            # Single factor used for all modes
            X = X - stepsize * preconditioned_grad.reshape(X.shape)
            Y = X
            Z = X
        else:
            # Split the preconditioned gradient into 3 parts
            prgx, prgy, prgz = split_fn(preconditioned_grad, sizes)
            X = X - stepsize * prgx
            Y = Y - stepsize * prgy
            Z = Z - stepsize * prgz
    else:
        # 2D Matrix
        if symmetric:
            # Single factor used for both rows/cols
            X = X - stepsize * preconditioned_grad.reshape(X.shape)
            Y = X
            Z = X  # Not strictly used in a matrix scenario, but kept for consistency
        else:
            # Split the preconditioned gradient into 2 parts
            prgx, prgy = split_fn(preconditioned_grad, sizes)
            X = X - stepsize * prgx
            Y = Y - stepsize * prgy

    return X, Y, Z


def rebalance(X, Y, device):
    """
    Rebalances matrices X and Y using QR decompositions and SVD, 
    with all computations performed on the specified device.
    
    Parameters:
        X (torch.Tensor): Input matrix X.
        Y (torch.Tensor): Input matrix Y.
        device (torch.device): The device to perform computations on.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rebalanced matrices L_OPSAd and R_OPSAd.
    """
    # Move input matrices to the specified device.
    X = X.to(device)
    Y = Y.to(device)
    
    # Perform reduced QR decompositions of X and Y.
    QL, WL = torch.linalg.qr(X, mode='reduced')
    QR, WR = torch.linalg.qr(Y, mode='reduced')
    
    # Compute the SVD of the product WL @ WR.T.
    U, s, Vh = torch.linalg.svd(WL @ WR.T)
    
    # Create a diagonal matrix of the square roots of the singular values.
    Sigma_sqrt = torch.diag(torch.sqrt(s)).to(device)
    
    # Compute the rebalanced matrices.
    L_OPSAd = QL @ U @ Sigma_sqrt
    R_OPSAd = QR @ Vh.T @ Sigma_sqrt
    
    return L_OPSAd, R_OPSAd



def matrix_inverse_sqrt(A, device, eps=1e-13):
    """
    Computes A^(-1/2) for a symmetric matrix A on a specified device.
    
    Parameters:
        A (torch.Tensor): A symmetric matrix of shape (N, N).
        device (torch.device): The device to run computations on.
        eps (float): A small constant for numerical stability.
    
    Returns:
        torch.Tensor: The matrix inverse square root of A, on the specified device.
    """
    # Ensure A is on the correct device.
    A = A.to(device)
    
    # Compute the eigenvalue decomposition of A.
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    
    # Clamp eigenvalues for numerical stability.
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    
    # Compute the inverse square root of the eigenvalues.
    inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)
    
    # Create a diagonal matrix from the inverse square roots.
    # Ensure it is created on the same device.
    inv_sqrt_diag = torch.diag(inv_sqrt_eigenvalues).to(device)
    
    # Reconstruct A^(-1/2) = Q * diag(inv_sqrt_eigenvalues) * Q^T.
    A_inv_sqrt = eigenvectors @ inv_sqrt_diag @ eigenvectors.T
    
    return A_inv_sqrt

def scaled_action(update, factors, symmetric, tensor=True):
    
    if tensor:
        raise NotImplementedError('Scaled does not exist for CP factorization')
    else:
        # Matrix factorization case.
        if symmetric:
            X = factors[0]
            g_mat = update.reshape(X.shape)
            return (2 * g_mat @ (X.T @ X)).reshape(-1)
        else:
            X, Y = factors
            shapes = [X.shape, Y.shape]
            gx, gy = split(update, shapes)
            op_x = (gx @ (Y.T @ Y)).reshape(-1)
            op_y = (gy @ (X.T @ X)).reshape(-1)
            return torch.cat((op_x, op_y))

from time import perf_counter
    
def ad_hoc_matrix_sensing(
    X, Y, Z,
    T_star,
    measurement_operator,
    y_observed,
    n_iter,
    method,
    loss_ord,
    m,
    n1, n2, n3,
    sizes,
    split,
    symmetric=False,
    tensor=False,
    geom_decay=None,
    q=None,
    lambda_=None,
    gamma=None,
    gamma_custom=None,
    device=None
):
    """
    Perform iterative optimization of factors X, Y, (and Z if tensor=True).
    Returns:
        errs: list of relative errors (length ≤ n_iter)
        X, Y, Z: the final updated factors
    """
    errs = []
    times = []
    t0 = perf_counter()

    for k in range(n_iter):
        # Reconstruct the model
        if tensor:
            T = torch.einsum('ir,jr,kr->ijk', X, Y, Z)
        else:
            T = X @ Y.T

        err = torch.norm(T - T_star)
        rel_err = err / torch.norm(T_star)
        if k % 20 == 0:
            print(f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}")

        errs.append(rel_err.item())
        times.append(perf_counter()- t0)

        # Compute residual and its subgradient
        residual = measurement_operator.A(T) - y_observed
        if loss_ord == 1:
            subgradient_h = measurement_operator.A_adj(torch.sign(residual)).view(-1)
            h_c_x = torch.sum(torch.abs(residual)).item()
        elif loss_ord == 0.5:
            subgradient_h = measurement_operator.A_adj(residual / torch.norm(residual)).view(-1)
            h_c_x = torch.norm(residual).item()
        elif loss_ord == 2:
            subgradient_h = measurement_operator.A_adj(residual).view(-1)
            h_c_x = 0.5 * (torch.norm(residual) ** 2).item()
        elif loss_ord == 10:
            subgradient_h = (1/m) * measurement_operator.A_adj(torch.sign(residual)).view(-1)
            h_c_x = (1/m) * torch.sum(torch.abs(residual)).item()
        else:
            raise ValueError(f"Unsupported loss_ord: {loss_ord}")

        # Compute (pre)gradient
        grad = compute_gradient(
            X, Y, Z if tensor else None,
            subgradient_h, n1, n2, n3,
            symmetric=symmetric, tensor=tensor
        )
        if method == 'OPSA($\\lambda=10^{-8}$)':
            grad += 1e-8 * torch.cat((X.reshape(-1), Y.reshape(-1)))

        # Stepsize & damping selection
        stepsize, damping = compute_stepsize_and_damping(
            method, grad, subgradient_h, h_c_x, loss_ord,
            symmetric, tensor=tensor,
            geom_decay=geom_decay, q=q,
            lambda_=lambda_, gamma=gamma,
            k=k, X=X, Y=Y,
            G=split(grad, sizes),
            device=device, gamma_custom=gamma_custom
        ) 

        # Build operator for CG
        factors = [X, Y, Z] if tensor else [X, Y]
        if method in ['Precond. gradient', 'Scaled gradient($\\lambda=10^{-8}$)', 'OPSA($\\lambda=10^{-8}$)']:
            operator_fn = lambda x: scaled_action(x, factors, symmetric, tensor=tensor)
        elif method in ['Gradient descent', 'Polyak Subgradient']:
            operator_fn = lambda x: x
        else:
            raise NotImplementedError(f"Unknown method: {method}")

        # Solve for (preconditioned) gradient update
        preconditioned_grad = cg_solve(operator_fn, grad, damping)

      

        # Update factor matrices/tensors
        X, Y, Z = update_factors(
            X, Y, (Z if tensor else None),
            preconditioned_grad if method not in ['Gradient descent', 'Polyak Subgradient'] else grad,
            stepsize, sizes, split,
            symmetric=symmetric, tensor=tensor
        )
        if method == 'OPSA($\\lambda=10^{-8}$)':
            X, Y = rebalance(X, Y, device)

    return errs,times

from collections import defaultdict
from typing import Dict, Tuple, Any

def flip_outputs(
    outputs: Dict[Tuple[int, float], Dict[str, Any]]
) -> Dict[str, Dict[Tuple[int, float], Any]]:
    """
    Swap the two nesting levels of an `outputs` dictionary.

    Original shape
    --------------
    outputs[(r, kappa)][method] = errs

    Desired shape
    -------------
    flipped[method][(r, kappa)] = errs

    Parameters
    ----------
    outputs : dict
        Keys are `(r, kappa)` tuples; values are dicts mapping
        `method` names to `errs`.

    Returns
    -------
    dict
        A new dictionary with the levels flipped.
    """
    flipped: Dict[str, Dict[Tuple[int, float], Any]] = defaultdict(dict)

    for (r, kappa), per_method in outputs.items():
        for method, errs in per_method.items():
            flipped[method][(r, kappa)] = errs

    return dict(flipped)  # convert back to a regular dict if desired