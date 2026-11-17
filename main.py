"""gta6-twitch-ccu-forecast-01, model C: diurnal shape + damped trend.

Idea
----
A launch day's Twitch curve is two things multiplied together:

* a diurnal component -- how many people are awake and watching at each
  UTC hour (the curve sags in the small hours, jumps at the US East
  midnight unlock, and peaks in the 19:00-22:00 UTC evening), and
* a damped launch trend -- the launch spike decays toward the game's
  steady level, and how far it has to fall depends on how much of the
  audience was already in at hour 1.

At run time the program fits, for every forecast hour h = 2..24
separately, a one-regressor line through the comparable past launches

    ln(viewers_h) = a_h + b_h * ln(viewers_1)

so the diurnal shape lives in the per-hour intercepts a_h and the damping
in the per-hour slopes b_h (b_h < 1: a bigger hour-1 print predicts a
flatter rest of the day; the evening slope is about 0.5, i.e. the evening
level grows only like sqrt(hour-1)).  The spread is the regression's
residual standard deviation per hour, and the five quantiles are the
normal quantiles around the fitted median.

Only comparables that look like this launch are allowed to vote.  A
donor must have a complete release day, must not have launched already
by the start of the day before release (its day-minus-1 00:00-05:00 UTC
mean is under 15% of that day's peak -- this throws out early-access
and day-early global-PC launches), must have had a real hour-1 audience
(at least 20,000 viewers), and its hour-1 print must be a substantial
share of its pre-launch day's peak (25%-105%: the launch was under way
at hour 1, but hour 1 was not already the peak).  The filters are
relaxed step by step if too few donors survive, and if no donor
survives at all the program falls back to a flat persistence forecast.

What was tried and dropped (measured on leave-one-out validation over
the comparables and on the four practice cases, whose answers were
re-fetched from SullyGnome):
* the case's own pre-launch diurnal profile (log viewers by UTC hour
  over the 14 pre-launch days) as a per-hour adjustment -- its
  correlation with the regression residuals is weak (about 0.0-0.35)
  and it does not lower leave-one-out pinball loss, so it is not used;
  the diurnal shape is carried by the per-hour intercepts instead;
* day-minus-1 peak / evening viewers and hour-1 channels as extra
  regressors -- no measurable gain over ln(viewers_1) alone;
* kernel weighting by similarity in ln(viewers_1) and k-nearest
  neighbours -- worse on leave-one-out.

Validation (see REPORT.md): practice pinball loss 0.171 against the
house baseline's 0.387; leave-one-out inside the donor pool 0.277.

Standard library only, no network, deterministic: the only inputs are
the case file's numbers, the fits are closed-form least squares, and
the output bytes are a pure function of the input.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
SIGMA_MULT = 0.85
MIN_DONORS = 8


def parse_time(text):
    """An ISO-8601 UTC timestamp -> datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def hour_offsets(hourly, day_start):
    """A comparable's hourly points -> {hour offset from day_start: viewers}."""
    out = {}
    for point in hourly or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        try:
            offset = int((parse_time(point["from"]) - day_start).total_seconds() // 3600)
        except (KeyError, TypeError, ValueError):
            continue
        out[offset] = float(viewers)
    return out


def donor_features(viewers_by_offset):
    """Pre-launch shape of one comparable, or None if it cannot be a donor.

    Returns (hour-1 viewers, day-minus-1 peak, day-minus-1 00:00-05:00 mean).
    A donor needs a complete release day (offsets 0..23 all present).
    """
    if any(h not in viewers_by_offset for h in range(24)):
        return None
    day_before = [v for off, v in viewers_by_offset.items() if -24 <= off < 0]
    early = [v for off, v in viewers_by_offset.items() if -24 <= off < -18]
    if not day_before or not early:
        return None
    peak = max(day_before)
    if peak <= 0:
        return None
    return viewers_by_offset[0], peak, sum(early) / len(early)


def select_donors(donors):
    """Donors that look like a rolling-midnight console launch.

    donors: list of (viewers_by_offset, features).  The filters relax
    step by step so the program always has donors when any exist.
    """
    def passes(features, max_ratio, need_big_hour1, need_quiet_start):
        v1, peak, early_mean = features
        if need_quiet_start and early_mean / peak >= 0.15:
            return False
        if need_big_hour1 and v1 < 20000:
            return False
        ratio = v1 / peak
        return 0.25 <= ratio <= max_ratio

    stages = [
        (1.05, True, True),
        (1.05, False, True),
        (1.30, False, True),
        (1.30, False, False),
    ]
    for max_ratio, need_big_hour1, need_quiet_start in stages:
        kept = [d for d in donors if passes(d[1], max_ratio, need_big_hour1, need_quiet_start)]
        if len(kept) >= MIN_DONORS:
            return kept
    kept = [d for d in donors]
    return kept


def fit_line(xs, ys):
    """Closed-form least squares y = a + b*x; returns (a, b, resid_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.5
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    sd = math.sqrt(sse / max(1, n - 2)) if n > 2 else 0.5
    return a, b, sd


def forecast_case(case):
    """One case -> {"hours": [{"hour": h, "q": [...]}, ...]} for h = 2..24."""
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    else:
        hourly = case.get("history_hourly") or []
        p1 = max(1.0, float(hourly[-1]["viewers"])) if hourly else 1.0

    donors = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        by_offset = hour_offsets(comp.get("hourly"), day_start)
        features = donor_features(by_offset)
        if features is not None:
            donors.append((by_offset, features))

    hours = []
    if len(donors) >= 2:
        pool = select_donors(donors)
        log_v1 = math.log(p1)
        for row in case["forecast"]:
            h = row["hour"]
            xs = [math.log(d[0][0]) for d in pool]
            ys = [math.log(d[0][h - 1]) for d in pool]
            a, b, sd = fit_line(xs, ys)
            mu = a + b * log_v1
            sigma = max(0.12, sd) * SIGMA_MULT
            qs = []
            prev = 0.0
            for z in Z:
                q = math.exp(mu + z * sigma)
                q = min(max(q, 1.0), 1e9)
                if q < prev:
                    q = prev
                qs.append(round2(q))
                prev = q
            hours.append({"hour": h, "q": qs})
    else:
        # No usable donor at all: flat persistence with a widening spread.
        for row in case["forecast"]:
            h = row["hour"]
            sigma = 0.15 * math.sqrt(h - 1)
            qs = [round2(min(max(p1 * math.exp(z * sigma), 1.0), 1e9)) for z in Z]
            hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
