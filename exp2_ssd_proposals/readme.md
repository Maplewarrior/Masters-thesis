## Goal
In this experiment we want to investigate the behavior of the SSD method and how the choice of hyperparameters affect the decision boundary. We also try to address the shortcomings of SSD in two ways:

* 1) By applying dampening at only the last layer of the model. This is referred to as "SSD v2" 
* 2) By searching for hyperparameters whose dampening yields a loss as close as possible to the original model's loss on a "generalization dataset" constructed in the same way as the repair step for Scrub+R. This is referred to as "SSD v3".
* 3) A combination of the previous two approaches. This is reffered to as "SSD v4".
* 4) Here we learn two sets of parameters: $(\alpha_1, \lambda_1)$ which are applied to the first half of the network and $(\alpha_2, \lambda_2)$ which are applied to the second half. Additionally, when constructing the generalization set we check if all classes are present in forget. If not, additional points for the missing classes are added BOTH to the forget set and the generalization set. We hypothesize that this will favor parameters that do not induce catastrophic forgetting.


## Creating the data
First, make sure you have created the data for the rogue experiments. This is done in experiment 1. 

Run either: 
```bash
python exp1_decision_boundary/generate_data_rogue_many.py
python exp1_decision_boundary/generate_data_rogue_one.py
```
This will generate the datasets including plots in your repo root directory's data folder. The data is saved in the `data/rogue_many` folder or `data/rogue_one` folder. There are 5 different cases of data, which means you can reference the data by `data/rogue_many/data_1.npz`, `data/rogue_many/data_2.npz`, etc. in your `config.yaml` when running experiments.


## Running experiments
Now we run the experiment on each of the dataset configuration, moving the rogue point(s).

```bash
python exp2_rogue_many/experiment_run.py data.dataset="data/rogue_many/data_1.npz,data/rogue_many/data_2.npz,data/rogue_many/data_3.npz,data/rogue_many/data_4.npz,data/rogue_many/data_5.npz" unlearn.method="ssd,ssd_v2,ssd_v3,ssd_v6,ssd_v6_smooth" hydra/launcher=ray --multirun
```