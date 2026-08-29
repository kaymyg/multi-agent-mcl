# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-07-28

### Added
- Initial reference implementation: `ProductionSimulationEnvironment`,
  `ContinuousGameEngine`, `AdvancedAdversarialSecurityModule`,
  `ProductionRiskAndAttributionController`, `ProductionMCLOrchestrator`.
- 13-stage tiered escalation pipeline (screening → deep audit → statistical
  governance gate), with non-parametric bootstrap threshold-breach testing
  and multivariate Gaussian root-cause attribution.
- Reference verification trace (`main.py`) and unit test suite.
- CI: install and `pytest -q` on every push and PR.

### Fixed
- Closed linguistic-anomaly and sub-threshold pacing evasion paths in
  `security_monitor.py`, `orchestrator.py`, and `risk_controller.py`.
- Fixed an unreachable quarantine gate and miscalibrated attribution: the
  utility function's `R_bonus(c)` term was omitted from the payoff
  evaluator, which pinned collusion at exactly `0.0` for an entire run and
  made the `STRATEGIC_DECEPTIVE_COLLUSION` profile — and the documented
  `QUARANTINE_ENGAGED` outcome — unreachable by construction.
- Test coverage for `main.py` and `simulation.py`, previously untested (0%).
- Dropped unused `torch`/`transformers` dependencies — nothing in the
  codebase imports them.
- CI: run via `python -m pytest` so the repo root is reliably on `sys.path`.

### Notes
- This is the first tagged release. All commits above predate any git tag;
  they're grouped here as the 1.0.0 baseline since that's what the working,
  currently-passing state of the repo represents.
- The README's "Actual output" example was regenerated against this version
  to replace stale numbers (`0.9700` / `84.XX%`) left over from before the
  quarantine-gate fix above — the real, reproducible output is
  `0.9950` / `97.49%`.
