---
name: Evasion report (non-sensitive)
about: A policy trajectory that should trigger quarantine but doesn't, and it's safe to discuss publicly
title: "[Evasion] "
labels: evasion
---

> If this is a working evasion you'd rather not publish before there's a
> fix, please use private vulnerability reporting instead (Security tab →
> "Report a vulnerability") per [SECURITY.md](../../SECURITY.md).

Use this template for evasions you're comfortable discussing openly —
theoretical gaps, edge cases already partially mitigated, or things that
overlap with a documented limitation.

**Policy vector / trajectory**
The `(e, c, h)` values or sequence of turns that should have triggered
`TIER_2_STABILIZED` or `QUARANTINE_ENGAGED` but didn't.

**Which stage should have caught it?**
Reference the 13-stage pipeline in the README.

**Minimal reproduction**
Ideally as a new test case, in the style of the existing evasion-path tests.

**Is this close to an existing attribution profile?**
`STRATEGIC_DECEPTIVE_COLLUSION`, `MEMORY_VECTOR_CORRUPTION`, or something the
current two profiles don't cover?
