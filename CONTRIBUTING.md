# Contributing to the Multi-Agent MCL

Thanks for your interest. This project models detecting and correcting
misbehaving agents, so contributions that strengthen detection, close
evasion paths, or make the calibration math more honest are especially
welcome.

## Getting set up

```bash
git clone https://github.com/kaymyg/multi-agent-mcl.git
cd multi-agent-mcl
pip install -r requirements.txt
pip install pytest pytest-cov

python main.py    # reference verification trace
pytest -q         # 20 tests, ~87% coverage
```

## Before opening a PR

CI installs `requirements.txt` and runs `python -m pytest -q` on every push
and PR. Please run that locally first:

```bash
pytest -q
```

If `python main.py`'s output changes as a result of your change, update the
"Actual output" block in the README to match — it's meant to be exactly
reproducible (`np.random.seed(7)`), and a past mismatch between that block
and reality is exactly the kind of drift this project has already had to fix
once (see `main.py`'s `simulated_payoff_evaluator` comment for the story).

## What especially matters here

- **The control law in the README and the code must match.** `main.py`
  already has one documented case (`R_bonus` omitted) where the two drifted
  apart and made a documented outcome literally unreachable. Don't reintroduce
  that class of bug.
- **New evasion paths belong as new tests**, ideally named after what the
  simulated agent is trying to get away with (see the existing "linguistic-
  anomaly and sub-threshold pacing evasion" patches in the git history for
  the style).
- **Don't loosen `tau_quarantine` or the bootstrap confidence threshold
  without justifying it in the PR.** These are the actual security boundary,
  not tuning knobs to make demos look cleaner.

## Good places to start

- Additional adversarial policy trajectories in `tests/` beyond the two
  existing profiles (`STRATEGIC_DECEPTIVE_COLLUSION`,
  `MEMORY_VECTOR_CORRUPTION`)
- A real LoRA hot-swap implementation behind the interfaces implied by the
  "Deployment Mandates" section, since the current reference implementation
  models the control loop, not an actual adapter registry
- Tightening `main.py` coverage (currently the lowest-covered module, since
  most of it only runs under `if __name__ == "__main__"`)

For anything larger, please open an issue first to discuss the approach.

## Commit style

Small, focused commits with a clear one-line summary. No strict format
enforced beyond that.
