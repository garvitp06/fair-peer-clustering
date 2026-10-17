"""Fast 0/1 programs: LP relaxation, then an exact repair of the fractional part.

Fair assignment programs (Bera et al. 2019) have LP relaxations whose optimal
vertices are almost integral: only a handful of points are split between
clusters. We therefore

1. solve the LP relaxation with HiGHS,
2. fix every variable that is already 0 or 1,
3. solve the remaining (small) integer program exactly,

and fall back to a full MILP with a 1% gap if the repair is infeasible. This is
orders of magnitude faster than a cold MILP while returning an assignment that
satisfies every constraint exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_binary(c: np.ndarray, cons: LinearConstraint, time_limit: float = 60.0,
                 owner: np.ndarray | None = None):
    """Minimise c @ x over x in {0,1}^n subject to ``cons``. Returns x or None.

    ``owner`` optionally maps each variable to an entity (e.g. the point being
    assigned). If any variable of an entity is fractional, all of that entity's
    variables are freed in the repair step, giving it full flexibility.
    """
    n = len(c)
    lp = milp(c=c, constraints=cons, integrality=np.zeros(n), bounds=Bounds(0, 1),
              options={"time_limit": time_limit, "disp": False})
    if lp.x is None:
        return None
    x = lp.x
    frac = (x > 1e-6) & (x < 1 - 1e-6)
    if not frac.any():
        return np.round(x)
    if owner is not None:
        frac = np.isin(owner, np.unique(owner[frac]))
    lb = np.where(frac, 0.0, np.round(x))
    ub = np.where(frac, 1.0, np.round(x))
    rep = milp(c=c, constraints=cons, integrality=np.ones(n), bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": False})
    if rep.x is not None:
        return np.round(rep.x)
    full = milp(c=c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
                options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 0.01})
    return None if full.x is None else np.round(full.x)
