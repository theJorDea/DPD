# Experiment execution ledger

> **Historical first-stage snapshot.** The status below predates the
> two-loop PA-identification/GMP work and must not be used as the current
> project status, acceptance protocol or execution queue.

The normative current protocol is
[../EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md), the current sequencing is
[../ROADMAP.md](../ROADMAP.md), and completed forward-PA evidence is summarized
in [../PA_MODEL_BENCHMARK.md](../PA_MODEL_BENCHMARK.md). This file is retained
only to explain which first-stage surrogate-only jobs were executed. In
particular, “28 unit tests” is a historical count; the later recorded suite is
131/131.

## Completed

- source/data provenance and three repository audits;
- NumPy evaluator primitives and 28 unit tests;
- DPA_200MHz 280-candidate inverse sweep;
- DPA_200MHz 280-candidate train-only memory-polynomial-surrogate sweep;
- APA_200MHz 280-candidate train-only memory-polynomial-surrogate sweep;
- frozen test evaluation for both datasets in `experiments/results/`;
- fixed-point numerical emulation on the selected spline models;
- CPU reproduction of Egor's four EnhancedESN_FAN fits (reported separately).

## Not completed

- local OpenDPD neural reproduction (no GPU/checkpoints in checkout);
- physical PA closed-loop measurement;
- spline memory branches/OMP/group-LASSO;
- three-seed neural confidence intervals;
- hardware latency/throughput or bit-true RTL.

Every result directory contains the training report, validation trial ledger,
selected NPZ, surrogate NPZ where applicable, and frozen test JSON/NPZ. Values
are labelled `surrogate_only` unless a physical measurement is explicitly
identified.

The later GMP workflow lives in separate
`experiments/results/pa_gmp_*_{selection,residuals,test}` directories. Its
release-gate PASS authorized one frozen forward-model test per dataset; it did
not pass Gate A→B and did not convert any result above into physical-PA DPD
evidence.
