# Experiment 6: Teacher Ascend Decision Boundaries

## Generating data

```bash
python exp1_decision_boundary/generate_data_rogue_many.py
python exp1_decision_boundary/generate_data_rogue_one.py
```

## Running experiments

```bash
python experiment_run.py data.dataset="data/rogue_many/data_1.npz,data/rogue_many/data_2.npz,data/rogue_many/data_3.npz,data/rogue_many/data_4.npz,data/rogue_many/data_5.npz" --multirun
```