# -*- coding: utf-8 -*-
#Author: Abdel Ghani Labassi
import torch
from utils import split 

def GN_action_matrix_tensor(update, factors, symmetric, tensor=True):
    """
    Computes the action of the linearized operator (nabla c.T * nabla c) for the CP factorization map (or symmetric/asymmetric matrix)
    on a search direction “update”. This is used inside the conjugate–gradient solver.

    Args:
        update (torch.Tensor): The update direction (flattened)
        factors
    Returns:
        torch.Tensor: The result of applying the operator.
    """
    if tensor:
        if symmetric:
            X = factors[0]
            dX = update.reshape(X.shape)
            XX = X.T @ X
            return (3 * (dX @ (XX * XX)) + 6 * (X @ ((dX.T @ X) * XX))).reshape(-1)
        else:
            X, Y, Z = factors
            shapes = [X.shape, Y.shape, Z.shape]
            dX, dY, dZ = split(update, shapes)
            XX = X.T @ X
            YY = Y.T @ Y
            ZZ = Z.T @ Z
            RES_X = dX @ (YY * ZZ) + X @ ((dY.T @ Y) * ZZ) + X @ (YY * (dZ.T @ Z))
            RES_Y = dY @ (XX * ZZ) + Y @ ((dX.T @ X) * ZZ) + Y @ (XX * (dZ.T @ Z))
            RES_Z = dZ @ (XX * YY) + Z @ ((dX.T @ X) * YY) + Z @ (XX * (dY.T @ Y))
            return torch.cat((RES_X.reshape(-1), RES_Y.reshape(-1), RES_Z.reshape(-1)))
    else:
        # Matrix factorization case.
        if symmetric:
            X = factors[0]
            g_mat = update.reshape(X.shape)
            return (2 * g_mat @ (X.T @ X) + 2 * X @ (g_mat.T @ X)).reshape(-1)
        else:
            X, Y,_ = factors
            shapes = [X.shape, Y.shape]
            gx, gy = split(update, shapes)
            op_x = (X @ (gy.T @ Y) + gx @ (Y.T @ Y)).reshape(-1)
            op_y = (Y @ (gx.T @ X) + gy @ (X.T @ X)).reshape(-1)
            return torch.cat((op_x, op_y))


