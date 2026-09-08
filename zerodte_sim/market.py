"""Intraday path generation and the 0DTE implied-vol surface.

The model is built around a **variance clock** rather than wall-clock time.
``W(u)`` is the fraction of the session's total variance realised by time
fraction ``u``, and it is U-shaped: the open and the last hour carry far more
variance per minute than midday.  Both the simulated path and the option
pricing read from the same clock, so a spread's decay and the underlying's
motion stay consistent with each other.

Three ingredients drive whether a premium seller makes money:

1. **The variance risk premium.** Realised daily vol is drawn as a multiple of
   what was implied at the open.  The multiplier's median is below 1 (that is
   the seller's edge) but its distribution has a long right tail (that is how
   the edge gets paid back).  This is the single most important parameter in
   the whole simulator -- see ``MarketConfig.vrp_median``.
2. **Trend versus chop.** A fraction ``theta`` of the day's variance is
   delivered as a one-directional drift and the rest as noise.  Total variance
   is identical either way, but a trend day walks through your short strike and
   keeps going, while a chop day with the same realised vol never touches it.
   This is what makes rolling path-dependent instead of just size-dependent.
3. **The leverage effect.** Implied vol expands when the market falls.  A
   losing put spread therefore costs more to close than its realised move alone
   implies, and the replacement spreads sold after the drop pay a richer credit.
   Both effects act directly on the roll decision, in opposite directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pricing import bs_delta, bs_price, intrinsic, strike_for_delta

__all__ = ["MarketConfig", "DaySurface", "MarketSimulator"]

MINUTES_PER_SESSION = 390  # 09:30 -> 16:00 ET


@dataclass
class MarketConfig:
    """Parameters of the price/vol process.  All vols are *daily* fractions."""

    spot: float = 6000.0

    # --- level -----------------------------------------------------------
    implied_daily_move: float = 0.0080
    """ATM implied one-day move at the open, as a fraction of spot.

    0.0080 on a 6000 index is a ~48pt expected move and a ~38pt straddle,
    which is a typical quiet-to-normal SPX session.
    """

    # --- variance risk premium ------------------------------------------
    vrp_median: float = 0.78
    """Median of realised / implied daily *vol*.

    Careful: the median is not the edge.  Because the multiplier is lognormal,
    raising ``vrp_log_sd`` inflates mean realised *variance* even with the
    median pinned.  The quantity that decides whether a premium seller has an
    edge is :attr:`mean_variance_ratio`; check it after changing either knob.
    """

    vrp_log_sd: float = 0.42
    """Dispersion of the realised/implied multiplier (lognormal sigma).  Drives
    the fat right tail -- the days that pay back a month of credit."""

    # --- path shape ------------------------------------------------------
    trend_alpha: float = 1.1
    trend_beta: float = 3.0
    """Beta(alpha, beta) for the fraction of daily variance delivered as
    directional drift.  Mean ~0.27: most days are choppy, a minority trend."""

    reversal_amp: float = 0.6
    """Scale of the intraday round-trip excursion, in units of the day's vol.

    Without this the directional component holds one sign all session, and a
    day that trends down never comes back.  That single assumption is enough to
    make "sell the untested side after a selloff" look like free money, because
    the model never delivers the V-shaped reversal that is the whole risk of
    that trade.  The excursion is a tent peaking at a random point in the
    session and returning to zero by the close, so it lengthens the *path*
    without touching where the day *ends* -- terminal variance, and therefore
    the calibrated variance risk premium, are unaffected."""

    autocorr: float = 0.05
    """AR(1) coefficient on the standardised noise increments.  Adds short-term
    momentum without changing total variance."""

    jump_prob: float = 0.02
    jump_sd: float = 0.010
    """Poisson intraday shock: probability per day, and the scale of the jump
    as a fraction of spot."""

    # --- vol surface -----------------------------------------------------
    skew_slope: float = 0.055
    skew_curv: float = 0.015
    """IV as a function of standardised log-moneyness ``z``:
    ``iv(z) = level * (1 - slope*z + curv*z**2)``.  Positive slope lifts OTM
    puts (z < 0) above OTM calls -- the equity smirk."""

    lev_beta_down: float = 0.25
    lev_beta_up: float = 0.10
    """Leverage effect.  A one-sigma *down* move multiplies the IV level by
    ``1 + lev_beta_down``; a one-sigma up move divides it by roughly
    ``1 + lev_beta_up``."""

    iv_mult_cap: float = 3.0

    # --- variance clock --------------------------------------------------
    clock_open_amp: float = 1.6
    clock_close_amp: float = 1.1
    clock_scale: float = 0.10
    """Intensity ``lambda(u) = 1 + open_amp*exp(-u/s) + close_amp*exp(-(1-u)/s)``
    normalised to integrate to 1.  U-shaped intraday volatility."""

    step_minutes: int = 5

    def __post_init__(self) -> None:
        if MINUTES_PER_SESSION % self.step_minutes:
            raise ValueError(
                f"step_minutes={self.step_minutes} must divide {MINUTES_PER_SESSION}"
            )
        if not 0.0 < self.implied_daily_move < 1.0:
            raise ValueError("implied_daily_move must be a fraction in (0, 1)")
        if self.vrp_median <= 0.0:
            raise ValueError("vrp_median must be positive")

    @property
    def mean_variance_ratio(self) -> float:
        """``E[realised variance] / implied variance`` for the full model.

        This is the honest summary of the seller's edge.  Below 1.0 the variance
        risk premium is positive and short premium is paid to take it; at 1.0
        the strategy is a coin flip before costs; above 1.0 it bleeds no matter
        how the trade is managed.  Around 0.90 is a reasonable base case for
        SPX, and re-running at 1.0 is the stress test worth doing.

        Includes the jump contribution, which is easy to forget and is worth
        several ratio points on its own.
        """
        diffusive = self.vrp_median ** 2 * math.exp(2.0 * self.vrp_log_sd ** 2)
        jump = self.jump_prob * self.jump_sd ** 2 / self.implied_daily_move ** 2
        return diffusive + jump

    @property
    def n_steps(self) -> int:
        """Number of intervals in the session (grid has ``n_steps + 1`` points)."""
        return MINUTES_PER_SESSION // self.step_minutes

    def minute_to_index(self, minute: int) -> int:
        """Grid index at or after ``minute`` (0 = 09:30, 390 = 16:00)."""
        idx = math.ceil(minute / self.step_minutes)
        return min(max(idx, 0), self.n_steps)

    def variance_clock(self) -> np.ndarray:
        """Cumulative variance fraction ``W`` on the step grid, ``W[0] = 0``."""
        # Integrate lambda on a fine grid so the shape is resolved regardless of
        # how coarse step_minutes is, then sample at the step boundaries.
        sub = 20
        fine = np.linspace(0.0, 1.0, self.n_steps * sub + 1)
        lam = (
            1.0
            + self.clock_open_amp * np.exp(-fine / self.clock_scale)
            + self.clock_close_amp * np.exp(-(1.0 - fine) / self.clock_scale)
        )
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1])) ])
        cum /= cum[-1]
        return cum[::sub].copy()


class DaySurface:
    """One simulated session: the price path plus the vol surface along it.

    Instances are cheap views over pre-generated numpy arrays; the strategy
    engine indexes them by step and works in scalars from there.
    """

    __slots__ = (
        "spot_path", "iv_level", "var_left", "cfg", "n_steps",
        "vol_multiple", "vol_z", "signal_noise",
    )

    def __init__(
        self,
        spot_path: np.ndarray,
        iv_level: np.ndarray,
        var_left: np.ndarray,
        cfg: MarketConfig,
        vol_multiple: float = 1.0,
        vol_z: float = 0.0,
        signal_noise: float = 0.0,
    ) -> None:
        self.spot_path = spot_path
        self.iv_level = iv_level
        self.var_left = var_left
        self.cfg = cfg
        self.n_steps = len(spot_path) - 1
        self.vol_multiple = vol_multiple
        """Realised / implied daily vol for this session.  Not observable at
        entry time -- filters that read it are oracles, useful only as bounds."""
        self.vol_z = vol_z
        """``vol_multiple`` as a standard normal score."""
        self.signal_noise = signal_noise
        """Independent draw used to build a noisy observation of ``vol_z``.

        Held on the day rather than folded into a signal so that filter quality
        can be swept without regenerating markets -- the same day yields a
        correlated family of signals, one per assumed predictive power."""

    def spot(self, step: int) -> float:
        return float(self.spot_path[step])

    def total_var(self, strike: float, step: int) -> float:
        """Total remaining variance for ``strike``, including skew."""
        remaining = float(self.var_left[step])
        if remaining <= 0.0:
            return 0.0
        level = float(self.iv_level[step])
        atm_var = level * level * remaining
        atm_sd = math.sqrt(atm_var)
        spot = float(self.spot_path[step])
        z = math.log(strike / spot) / atm_sd
        z = min(max(z, -4.0), 4.0)
        mult = 1.0 - self.cfg.skew_slope * z + self.cfg.skew_curv * z * z
        mult = min(max(mult, 0.5), 3.0)
        return atm_var * mult * mult

    def price(self, strike: float, step: int, is_call: bool) -> float:
        if step >= self.n_steps:
            return intrinsic(self.spot(step), strike, is_call)
        return bs_price(self.spot(step), strike, self.total_var(strike, step), is_call)

    def delta(self, strike: float, step: int, is_call: bool) -> float:
        return bs_delta(self.spot(step), strike, self.total_var(strike, step), is_call)

    def strike_for_delta(
        self, step: int, target_delta: float, is_call: bool, iters: int = 12
    ) -> float:
        """Skew-consistent strike for a target absolute delta.

        The flat-vol inversion is exact, but the right vol depends on the strike
        we are solving for, so iterate the two to a fixed point.  Converges in a
        handful of passes for any sane surface.
        """
        spot = self.spot(step)
        remaining = float(self.var_left[step])
        if remaining <= 0.0:
            return spot
        strike = spot
        for _ in range(iters):
            var = self.total_var(strike, step)
            new_strike = strike_for_delta(spot, var, target_delta, is_call)
            if abs(new_strike - strike) < 1e-7 * spot:
                strike = new_strike
                break
            strike = new_strike
        return strike


class MarketSimulator:
    """Vectorised generator of independent 0DTE sessions."""

    def __init__(self, cfg: MarketConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self._clock = cfg.variance_clock()
        self._dw = np.diff(self._clock)
        self._noise_inflation = self._compute_noise_inflation(cfg.autocorr)

    def _compute_noise_inflation(self, rho: float) -> float:
        """Variance multiplier that AR(1) momentum applies to the terminal move.

        Summing ``sqrt(dw_i) * eps_i`` over an autocorrelated ``eps`` does not
        give total variance ``sum(dw) = 1``: the positive cross-covariances add
        to it.  At rho=0.05 over 78 steps that is a ~10% variance overshoot --
        more than the entire variance risk premium we are trying to model -- so
        divide it back out and let ``autocorr`` shape the path only, exactly as
        ``trend_alpha``/``trend_beta`` do.
        """
        if not rho:
            return 1.0
        a = np.sqrt(self._dw)
        lag = np.abs(np.arange(a.size)[:, None] - np.arange(a.size)[None, :])
        return float(a @ (rho ** lag) @ a)

    def generate(self, n_days: int) -> list[DaySurface]:
        cfg = self.cfg
        rng = self.rng
        n_steps = cfg.n_steps

        # --- how much the day actually moves --------------------------------
        vrp = cfg.vrp_median * np.exp(rng.normal(0.0, cfg.vrp_log_sd, n_days))
        realised_vol = cfg.implied_daily_move * vrp
        realised_var = realised_vol ** 2

        # --- split that variance into drift and noise -----------------------
        theta = rng.beta(cfg.trend_alpha, cfg.trend_beta, n_days)
        drift_sign = rng.choice([-1.0, 1.0], n_days)
        drift_total = drift_sign * realised_vol * np.sqrt(theta)
        noise_var = realised_var * (1.0 - theta)

        # Intraday round trip: peaks at a random point, back to zero by the bell.
        clock = self._clock[None, :]
        tau = rng.uniform(0.2, 0.8, n_days)[:, None]
        tent = np.where(clock <= tau, clock / tau, (1.0 - clock) / (1.0 - tau))
        excursion = (
            rng.standard_normal(n_days)[:, None]
            * cfg.reversal_amp
            * realised_vol[:, None]
            * tent
        )

        # --- correlated standardised innovations ----------------------------
        z = rng.standard_normal((n_days, n_steps))
        rho = cfg.autocorr
        if rho:
            eps = np.empty_like(z)
            eps[:, 0] = z[:, 0]
            keep = math.sqrt(1.0 - rho * rho)
            for t in range(1, n_steps):
                eps[:, t] = rho * eps[:, t - 1] + keep * z[:, t]
        else:
            eps = z

        dw = self._dw[None, :]
        directional = drift_total[:, None] * clock + excursion
        increments = np.diff(directional, axis=1) + np.sqrt(
            noise_var[:, None] * dw / self._noise_inflation
        ) * eps

        # --- intraday shocks -------------------------------------------------
        jumped = rng.random(n_days) < cfg.jump_prob
        if jumped.any():
            idx = np.flatnonzero(jumped)
            when = rng.integers(0, n_steps, idx.size)
            size = rng.standard_normal(idx.size) * cfg.jump_sd
            increments[idx, when] += size

        log_path = np.concatenate(
            [np.zeros((n_days, 1)), np.cumsum(increments, axis=1)], axis=1
        )
        spot_paths = cfg.spot * np.exp(log_path)

        # --- implied level responds to the path (leverage effect) -----------
        sigmas = log_path / cfg.implied_daily_move
        iv_mult = 1.0 + np.where(
            sigmas < 0.0, -cfg.lev_beta_down * sigmas, -cfg.lev_beta_up * sigmas
        )
        iv_mult = np.clip(iv_mult, 1.0 / cfg.iv_mult_cap, cfg.iv_mult_cap)
        iv_levels = cfg.implied_daily_move * iv_mult

        var_left = np.clip(1.0 - self._clock, 0.0, 1.0)

        vol_z = np.log(vrp / cfg.vrp_median) / cfg.vrp_log_sd
        signal_noise = rng.standard_normal(n_days)

        return [
            DaySurface(
                spot_paths[i], iv_levels[i], var_left, cfg,
                float(vrp[i]), float(vol_z[i]), float(signal_noise[i]),
            )
            for i in range(n_days)
        ]
