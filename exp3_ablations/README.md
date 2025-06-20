# Experiment 3: Ablation of BO-SSD

## Running the experiment

Run the experiment on `ssd-bo-pairwise` on all datasets:
```bash
python ablation_bo_objective.py unlearn.method=ssd-bo-pairwise data.dataset="data/rogue_many/data_1.npz,data/rogue_many/data_2.npz,data/rogue_many/data_3.npz,data/rogue_many/data_4.npz,data/rogue_many/data_5.npz" --multirun
```

Run the experiment on `ssd-bo-pairwise-smooth` on all datasets:
```bash
python ablation_bo_objective.py unlearn.method=ssd-bo-pairwise-smooth data.dataset="data/rogue_many/data_1.npz,data/rogue_many/data_2.npz,data/rogue_many/data_3.npz,data/rogue_many/data_4.npz,data/rogue_many/data_5.npz" --multirun
```

