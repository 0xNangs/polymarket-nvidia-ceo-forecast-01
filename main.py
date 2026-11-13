# polymarket-nvidia-ceo-forecast-01 entry: an adaptive forecaster.
#
# The house baseline holds each market's price flat with one fixed spread
# (0.0025 per square-root minute) for every market. These markets are not
# alike: some sit dead for hours, others jump twenty points in a minute.
# This program measures each market's own recent step size from the case's
# prices and sizes its bands to that market, widening like a random walk's.
#
# It reads every <id>.in in the input directory (the grader runs one case
# at a time) and writes <id>.out: {"forecasts":[{"id","q"}]}, q holding one
# row per horizon minute, each row the five quantiles 0.05/0.25/0.5/0.75/
# 0.95 of that minute's Yes price. Pure standard library, no randomness:
# two runs over the same case write the same bytes.

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
# The house baseline's per-level spread, in price units per square-root
# minute (its fixed 0.0025 times the normal quantile at each level). The
# fitted spreads below are blended toward these: the market's own history
# leads, the baseline steadies markets whose history is thin or wild.
BASE_C = (-0.004112, -0.001686, 0.0, 0.001686, 0.004112)
BLEND = 0.15

# Steps are measured on the last STEP_WINDOW prices only: a market's
# temper a day ago says little about its temper now.
STEP_WINDOW = 1440

# Horizons (minutes) at which the market's own step distribution is
# measured. For each, every overlapping step of that length in the case's
# prices is collected, and each level's empirical quantile of those steps
# is fitted against square-root-of-horizon through the origin: a random
# walk's spread grows that way, and the fit borrows strength across
# horizons instead of trusting any single one.
FIT_H = (5, 15, 30, 60, 120, 240)


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def _quantile(sorted_vals, tau):
    """The empirical quantile of a sorted list, linearly interpolated."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def fit_spreads(prices):
    """Per-level spread c_tau: the level's step quantile at horizon h is
    about c_tau * sqrt(h). Falls back to the baseline's spreads when the
    history is too short to measure."""
    series = prices[-STEP_WINDOW:]
    n = len(series)
    hs = [h for h in FIT_H if n - h >= 60]
    if not hs:
        return list(BASE_C)
    num = [0.0] * len(TAUS)
    den = 0.0
    for h in hs:
        steps = sorted(series[i + h] - series[i] for i in range(n - h))
        root = math.sqrt(h)
        den += h
        for j, tau in enumerate(TAUS):
            num[j] += root * _quantile(steps, tau)
    fitted = [num[j] / den for j in range(len(TAUS))]
    spreads = [
        (1.0 - BLEND) * fitted[j] + BLEND * BASE_C[j] for j in range(len(TAUS))
    ]
    # The median stays at the last price: the fitted step median is noise
    # around zero, and a wandering median pays at both sides. Persistence
    # was the baseline's one good idea; keep it.
    spreads[2] = 0.0
    return spreads


def forecast_market(prices, horizon):
    """horizon rows of the five quantiles for one market's next minutes."""
    x = prices[-1]
    c = fit_spreads(prices)
    rows = []
    for h in range(1, horizon + 1):
        root = math.sqrt(h)
        row = sorted(price6(x + c[j] * root) for j in range(len(TAUS)))
        rows.append(row)
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if tuple(case["quantiles"]) != TAUS:
            raise ValueError(f"{path.name} asks for quantiles {case['quantiles']}")
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], case["horizon_minutes"])}
            for m in case["markets"]
        ]
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
