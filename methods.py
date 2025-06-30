#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:19:02 2025

@author: aglabassi
"""
import torch
from time import perf_counter
import math


# ───────────────────────── basic helpers ──────────────────────────
def _init_timer():
    """Return zero-time reference and an empty timestamp list."""
    return perf_counter(), []


def _tick(t0, times):
    """Append elapsed wall-clock time (in seconds) to *times*."""
    times.append(perf_counter() - t0)


# ────────────────────────── core methods ──────────────────────────
def subgradient_method(
    stepsize_fc,
    subgradient_fc,
    action_nabla_F_transpose_fc,
    x0,
    n_iter=1000,
):
    xs, x = [], x0
    t0, times = _init_timer()

    for k in range(n_iter):
        xs.append(x)
        _tick(t0, times)

        α = stepsize_fc(k, x)
        g = action_nabla_F_transpose_fc(x, subgradient_fc(x))
        x = x - α * g

    xs.append(x)
    _tick(t0, times)
    return xs, times


def LM_subgradient_method(
    stepsize_fc,
    damping_fc,
    subgradient_fc,
    action_nabla_F_transpose_fc,
    levenberg_marquardt_linear_system_solver,
    x0,
    n_iter=1000,
    geom_decay=False
):
    xs, x = [], x0
    t0, times = _init_timer()

    for k in range(n_iter):
        xs.append(x)
        _tick(t0, times)
        λ = damping_fc(k, x)
        v =  subgradient_fc(x)
        g  = action_nabla_F_transpose_fc(x,v)
        pg = levenberg_marquardt_linear_system_solver(x, λ, g)
        
        if geom_decay:
            α = stepsize_fc(k, x)
        else:
            α = stepsize_fc(k, x, pg)
            
        x  = x - α * pg

    xs.append(x)
    _tick(t0, times)
    return xs, times


def GN_subgradient_method(
    stepsize_fc,
    subgradient_fc,
    action_nabla_F_transpose_fc,
    gauss_newton_linear_system_solver,
    x0,
    n_iter=1000,geom_decay=False
):
    xs, x = [], x0
    t0, times = _init_timer()

    for k in range(n_iter):
        xs.append(x)
        _tick(t0, times)
        g  = action_nabla_F_transpose_fc(x, subgradient_fc(x))
        pg = gauss_newton_linear_system_solver(x, g)
    
        if geom_decay:
            α = stepsize_fc(k, x)
        else:
            α = stepsize_fc(k, x, pg)
        
        x  = x - α * pg

    xs.append(x)
    _tick(t0, times)
    return xs, times