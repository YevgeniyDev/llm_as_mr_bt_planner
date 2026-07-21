# Immutable experiment snapshots

`scripts/run_experiment_matrix.py` creates a new timestamped directory here and
refuses to overwrite it. Each snapshot contains the exact matrix, commit SHA,
environment metadata, raw per-trial results, aggregate JSON/CSV, generated LaTeX
tables, and SHA-256 checksums. Commit the complete directory used by the paper.

LLM methods use the shared validator and blocking-guard simulator. MRBTP rows use
the explicitly labelled `mrbtp_native_v1` protocol because its Condition semantics
are different and cannot honestly be called same-simulator measurements.
