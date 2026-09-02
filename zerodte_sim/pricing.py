"""Black-Scholes for 0DTE index options, parameterised by total variance.

Two deliberate simplifications, both safe in this regime:

* **Zero rates and dividends.** With hours to expiry the carry term on an SPX
  option is worth a small fraction of a tick -- orders of magnitude below the
  bid/ask we already charge in :mod:`zerodte_sim.costs`.
* **Total variance instead of (sigma, T).**  Everything is expressed in
  ``v = sigma**2 * T``.  Annualised vol blows up as ``T -> 0``, which is the
  only regime this simulator ever visits; total variance decays smoothly to
  zero and the formulas stay numerically well behaved right into settlement.

Everything here is scalar pure-Python on purpose.  The inner loop prices a
handful of legs per step, and at that size ``math`` beats numpy's per-call
overhead by roughly an order of magnitude.
"""

from __future__ import annotations

import math
from statistics import NormalDist

__all__ = [
    "norm_cdf",
    "norm_ppf",
    "bs_price",
    "bs_delta",
    "strike_for_delta",
    "intrinsic",
]

_NORMAL = NormalDist()
_SQRT2 = math.sqrt(2.0)

# Below this the option is worth its intrinsic value to within a rounding error
# and the log/sqrt terms start losing precision.
_MIN_VAR = 1e-14


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def norm_ppf(p: float) -> float:
    """Standard normal quantile function."""
    return _NORMAL.inv_cdf(p)


def intrinsic(spot: float, strike: float, is_call: bool) -> float:
    """Value of the option at expiry."""
    return max(0.0, spot - strike) if is_call else max(0.0, strike - spot)


def bs_price(spot: float, strike: float, total_var: float, is_call: bool) -> float:
    """Undiscounted Black-Scholes price given total variance ``sigma**2 * T``."""
    if total_var <= _MIN_VAR:
        return intrinsic(spot, strike, is_call)
    sd = math.sqrt(total_var)
    d1 = (math.log(spot / strike) + 0.5 * total_var) / sd
    d2 = d1 - sd
    if is_call:
        return spot * norm_cdf(d1) - strike * norm_cdf(d2)
    return strike * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_delta(spot: float, strike: float, total_var: float, is_call: bool) -> float:
    """Spot delta.  Calls in [0, 1], puts in [-1, 0]."""
    if total_var <= _MIN_VAR:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    sd = math.sqrt(total_var)
    d1 = (math.log(spot / strike) + 0.5 * total_var) / sd
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0


def strike_for_delta(
    spot: float, total_var: float, target_delta: float, is_call: bool
) -> float:
    """Strike whose absolute delta equals ``target_delta``, at a flat vol.

    Inverting ``delta = Phi(d1)`` gives ``K = S * exp(v/2 - d1*sqrt(v))`` for
    both option types; only the sign of ``d1`` differs.  Callers that need the
    skew accounted for should iterate this against their vol surface -- see
    :meth:`zerodte_sim.market.DaySurface.strike_for_delta`.
    """
    if total_var <= _MIN_VAR:
        return spot
    target_delta = min(max(target_delta, 1e-6), 1.0 - 1e-6)
    sd = math.sqrt(total_var)
    d1 = norm_ppf(target_delta) if is_call else -norm_ppf(target_delta)
    return spot * math.exp(0.5 * total_var - d1 * sd)
