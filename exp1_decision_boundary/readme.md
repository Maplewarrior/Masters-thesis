# Experiment 1: Decision Boundary
This experiment explores the decision bondaries before and after unlearning for the rogue many and rogue one datasets on the following unlearning methods:
- Retrain
- Amnesiac
- SCRUB+R
- SSD
- SAE
- SISA

> [!NOTE]  
> The Amnesiac unlearning method does not have a repair step. Might not be necessary for this experiment as we are forgetting few points and points are easily separable.

> [!NOTE]  
> The SAE model is slightly different from the other models (it uses skip connections) and the results before unlearning will be different than the others, even when using the same seed.
> The same is the case for the SISA model as it is an ensemble of models.

## How to use

### Generating data
To generate the data, run:
```bash
python exp1_decision_boundary/generate_data_rogue_many.py
python exp1_decision_boundary/generate_data_rogue_one.py
```

This will generate the datasets including plots in your repo root directory's data folder. The data is saved in the `data/rogue_many` folder or `data/rogue_one` folder. There are 5 different cases of data, which means you can reference the data by `data/rogue_many/data_1.npz`, `data/rogue_many/data_2.npz`, etc. in your `config.yaml` when running experiments.


### Running experiments
To run the experiment for all the datasets and unlearning methods (for the rogue many datasets), run:
```bash
python exp1_decision_boundary/experiment_run.py data.dataset="data/rogue_many/data_1.npz,data/rogue_many/data_2.npz,data/rogue_many/data_3.npz,data/rogue_many/data_4.npz,data/rogue_many/data_5.npz" unlearn.method="amnesiac,ssd,retrain,scrubr,sae,sisa" hydra/launcher=ray --multirun
```

> [!TIP]
> The `hydra/launcher=ray` argument will run the experiments in parallel using Ray.

You can overwrite any parameters from the config file, for example `n_epochs`:
```bash
python exp1_rogue_many/experiment_run.py trainer.n_epochs=10
```
