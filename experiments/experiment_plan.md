# Experiment execution ledger

The normative protocol is [../EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md).
This file records what has actually been run in this workspace; it is kept
separate from the future physical-PA and GPU commands.

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
