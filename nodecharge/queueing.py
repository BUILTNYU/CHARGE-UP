"""Queueing reliability thresholds (Marianov and ReVelle, 1996).

Constraint (7) of Pantelidis et al. (2022) limits the demand assigned to a
node-charge (j, h) with m idle vehicles ("servers") to ``mu_jh * rho_m``, where
``rho_m`` is the largest utilisation for which an M/M/m queue keeps, with
probability at least ``eta``, no more than ``b`` customers waiting. ``rho_m``
is the binding root of Equation (19):

    sum_{k=0}^{m-1} (m - k) m! m^b / k! * (1 / rho)^(m + b + 1 - k) >= 1 / (1 - eta)
"""
from __future__ import annotations

from math import lgamma, log

import numpy as np
from scipy.optimize import brentq


def _log_lhs(rho: float, m: int, b: int) -> float:
    """log of the left-hand side of Eq. (19), evaluated stably for large m."""
    k = np.arange(m)
    terms = (
        np.log(m - k)
        + lgamma(m + 1)
        + b * log(m)
        - np.array([lgamma(x + 1) for x in k])
        - (m + b + 1 - k) * log(rho)
    )
    top = terms.max()
    return float(top + np.log(np.exp(terms - top).sum()))


def rho_threshold(m: int, eta: float = 0.95, b: int = 0) -> float:
    """Return the binding utilisation ``rho_m`` of Eq. (19) for ``m`` servers.

    Parameters
    ----------
    m : number of servers (idle vehicles) at a node-charge, m >= 1.
    eta : reliability level (probability of having at most ``b`` customers queued).
    b : allowed queue length.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must be in (0, 1)")
    log_target = -log(1.0 - eta)
    f = lambda r: _log_lhs(r, m, b) - log_target  # noqa: E731  (decreasing in r)
    lo, hi = 0.5, float(m)
    while f(lo) < 0:
        lo /= 2.0
    while f(hi) > 0:
        hi *= 2.0
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-12)


def rho_table(max_servers: int, eta: float = 0.95, b: int = 0) -> np.ndarray:
    """Return ``[0, rho_1, ..., rho_max_servers]`` (index = number of servers)."""
    return np.array([0.0] + [rho_threshold(m, eta, b) for m in range(1, max_servers + 1)])
