## Goal
In this experiment we want to investigate the behavior of the SSD method and how the choice of hyperparameters affect the decision boundary. We also try to address the shortcomings of SSD in two ways:

* 1) By applying dampening at only the last layer of the model. This is referred to as "SSD v2" 
* 2) By searching for hyperparameters whose dampening yields a loss as close as possible to the original model's loss on a "generalization dataset" constructed in the same way as the repair step for Scrub+R. This is referred to as "SSD v3".
* 3) A combination of the previous two approaches. This is reffered to as "SSD v4".
* 4) Here we learn two sets of parameters: $(\alpha_1, \lambda_1)$ which are applied to the first half of the network and $(\alpha_2, \lambda_2)$ which are applied to the second half. Additionally, when constructing the generalization set we check if all classes are present in forget. If not, additional points for the missing classes are added BOTH to the forget set and the generalization set. We hypothesize that this will favor parameters that do not induce catastrophic forgetting.


## Experiment runs
Now we run the experiment on each of the dataset configuration, moving the rogue point.

```bash
python exp2_rogue_many/experiment_run.py data.dataset="data/data_1.npz,data/data_2.npz,data/data_3.npz,data/data_4.npz,data/data_5.npz" unlearn.method="ssd,ssd_v2,ssd_v3,ssd_v4,ssd_v6,ssd_v6_smooth,ssd_v7,teacher_ascend" hydra/launcher=ray --multirun
```

### 1. Rogue point with same distance to all centroids (close to decision boundary)
<!-- ![Data configuration for Run 1](data/plots/data_1.png) -->
<img src="data/plots/data_1.png" alt="Dataset" width="600"/>

### Orignal model for the dataset
<!-- ![Original model for Run 1](results/1_decision_boundary_retrain_original.png) -->
<img src="results/1_decision_boundary_retrain_original.png" alt="Original Model" width="600"/>


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_v3_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_v4_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/1_decision_boundary_ssd_v5_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
</table>

### 2. Rogue point with same centroid as its class
<!-- ![Data configuration for Run 2](data/plots/data_2.png) -->
<img src="data/plots/data_2.png" alt="Dataset" width="600"/>

### Orignal model for the dataset
<!-- ![Original model for Run 2](results/2_decision_boundary_retrain_original.png) -->
<img src="results/2_decision_boundary_retrain_original.png" alt="Original Model" width="600"/>

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_v3_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_v4_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
      <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/2_decision_boundary_ssd_v5_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
<table>

### 3. Rogue point with same centroid as the class with a different label
<!-- ![Data configuration for Run 3](data/plots/data_3.png) -->
<img src="data/plots/data_3.png" alt="Dataset" width="600"/>

### Orignal model for the dataset
<!-- ![Original model for Run 3](results/3_decision_boundary_retrain_original.png) -->
<img src="results/3_decision_boundary_retrain_original.png" alt="Original Model" width="600"/>

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_v3_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_v4_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/3_decision_boundary_ssd_v5_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
<table>

### 4. Rogue point far away from its centroid, but probably in the same decision boundary
<!-- ![Data configuration for Run 4](data/plots/data_4.png) -->
<img src="data/plots/data_4.png" alt="Dataset" width="600"/>

### Orignal model for the dataset
<!-- ![Original model for Run 4](results/4_decision_boundary_retrain_original.png) -->
<img src="results/4_decision_boundary_retrain_original.png" alt="Original model" width="600"/>


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_v3_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_v4_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/4_decision_boundary_ssd_v5_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
<table>



### 5. Rogue point far away from its centroid, but probably in the same decision boundary
<!-- ![Data configuration for Run 5](data/plots/data_5.png) -->
<img src="data/plots/data_5.png" alt="Dataset" width="600"/>

### Orignal model for the dataset
<!-- ![Original model for Run 5](results/5_decision_boundary_retrain_original.png) -->
<img src="results/5_decision_boundary_retrain_original.png" alt="Original model" width="600"/>


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_v3_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_v4_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="600"/></td>
    <td><img src="results/5_decision_boundary_ssd_v5_unlearned.png" alt="Retrained Model" width="600"/></td>
  </tr>
  <tr>
<table>