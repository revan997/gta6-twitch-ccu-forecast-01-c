# GTA VI launch-day Twitch forecast — diurnal shape + damped trend

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/revan997/gta6-twitch-ccu-forecast-01-c/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/revan997/gta6-twitch-ccu-forecast-01-c/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

Entry for the [make.fun](https://make.fun/c/gta6-twitch-ccu-forecast-01)
competition `gta6-twitch-ccu-forecast-01`: forecast, hour by hour, how many
people watch Grand Theft Auto VI live on Twitch over the 24 UTC hours of
its release day, 2026-11-19, from the first hour's print.

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads one case file (`/input/<id>.in`) and writes `/output/<id>.out`:
five quantiles of the viewer count for each of hours 2–24.

The model treats a launch day's curve as a diurnal shape times a damped
launch trend. For every forecast hour separately it fits, through the
comparable past launches,

```
ln(viewers_h) = a_h + b_h * ln(viewers_1)
```

so the diurnal pattern lives in the per-hour intercepts `a_h` and the
damping in the per-hour slopes `b_h` (the evening slope is about 0.5 —
the evening level grows only like the square root of the hour-1 print).
The spread is the regression's residual standard deviation per hour,
and the five quantiles are the normal quantiles around the fitted median.

Only comparables that look like a rolling-midnight console launch are
allowed to vote: a complete release day, no launch already under way at
the start of the day before release, a real hour-1 audience, and an
hour-1 print that is a substantial share of the pre-launch day's peak.
Filters relax step by step if too few donors survive; with no donor at
all the program falls back to flat persistence.

## Validation

Practice answers were re-fetched from SullyGnome (the documented source)
and scored with the competition's own pinball loss on the log scale:

| model | practice loss | 90% coverage |
|---|---|---|
| house baseline (persistence) | 0.387 | 0.783 |
| this entry | **0.168** | 0.913 |

Leave-one-out over the 37 comparable launch days with the shipped code:
0.597 overall (the pool is a rolling-midnight filter, so it is the wrong
model for simultaneous-global-unlock and early-access launches), 0.283
within the donor pool. Per-case practice losses: p00 0.205, p01 0.151,
p02 0.131, p03 0.185. Runtime about 0.11 s per case; two runs are
byte-identical.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
