# LLMIP_basic (src_v1) - Input/Output Guide

## Overview
This project runs an LLM-based vaccination attitude simulation using workplace population and network data. It produces per-run outputs as well as aggregated statistics (mean/std) when running batch simulations.

## Inputs
### Required data files
Located under the project root:
- data/input/workplace_36013030400w1_extended_population.csv
- data/input/workplace_36013030400w1_extended_network.csv

### Configuration
Edit parameters in:
- src_v1/config.py

Key settings:
- MAX_STEPS: number of simulation steps per run
- BATCH_RUNS: number of runs for aggregation
- MAX_DIALOGS_PER_MICROSTEP: concurrent dialog count
- BELIEF_DISTRIBUTION_TYPE: neutral / pro / resist
- BELIEF_MEANS, BELIEF_STD: normal distribution parameters

## Outputs
Outputs are created under:
- data/output/simulation_<timestamp>/

When BATCH_RUNS = 1:
- run_01/ (per-run outputs)
- model_data_mean.csv (mean; same as run_01)
- model_data_std.csv (std; mostly zeros)

When BATCH_RUNS > 1:
- run_01/ ... run_NN/ (per-run outputs)
- model_data_mean.csv (mean across runs)
- model_data_std.csv (std across runs)

### Per-run files (inside each run_XX)
- model_data.csv: step-by-step model metrics
- agent_data.csv: per-agent metrics by step
- agent_profiles.json: agent profiles + belief histories
- all_dialogues.json: dialogue logs
- most_impactful_dialogues_report.txt: top dialogue summary
- network_data.json: network topology for visualization
- visualizations/ (charts and network frames)

### Key metrics in model_data.csv
- Vaccination_Rate
- Average_Belief_LLM
- Average_Belief_VADER
- Belief_Std_Dev_LLM
- Belief_Std_Dev_VADER

## Notes
- network_frames are generated from belief_history. If frames are missing, check that each step appends belief_history (already handled in src_v1/model.py).
- The aggregated mean/std files summarize final belief and vaccination metrics across runs.
