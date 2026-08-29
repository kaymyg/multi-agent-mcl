# Security Policy

## Scope

This is a reference implementation / architectural specification, not a
hardened production system. The simulation, game engine, and risk controller
model a control loop — they are not themselves a deployed multi-agent
runtime, so "vulnerability" here means something specific:

What **is** a genuine finding:

- A way for a simulated agent to reach `PERMIT_DEPLOYMENT` while its policy
  vector should trigger `TIER_2_STABILIZED` or `QUARANTINE_ENGAGED` under the
  documented thresholds (an evasion path)
- A mismatch between the control law described in the README and what the
  code actually computes, where the mismatch makes detection weaker (not
  just cosmetic)
- A way to falsify or manipulate the bootstrap confidence estimate or the
  root-cause attribution without the underlying policy vector actually
  changing
- A case where `TRIGGER_HARD_RESET_AND_WIPE` doesn't actually reset state
  used in the next evaluation, letting a quarantined agent's history leak
  forward

What is **not** in scope: this being "just a simulation" (it is, by design —
see the Deployment Mandates section of the README for what a real deployment
would need beyond this repo), or performance/scaling concerns.

## Reporting

Please **do not** open a public GitHub issue for a working evasion of the
containment gate — that publishes the exploit before there's a fix.

Instead, use GitHub's private vulnerability reporting for this repository
(Security tab → "Report a vulnerability"). Please include:

- The policy vector trajectory or test case that evades detection
- Which stage of the 13-stage pipeline should have caught it
- Whether it's a math/threshold issue or an implementation bug

For anything that's more "the architecture has a gap" than "the code has a
bug" — e.g. side channels not modeled at all — a public issue or discussion
is fine; that's design feedback, not a disclosure.

## Response expectations

Independent project — best-effort response, not a guaranteed SLA. Genuine
evasions will be prioritized and credited (with permission) once fixed.
