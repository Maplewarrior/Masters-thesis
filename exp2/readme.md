## Goal
In this experiment we want to investigate the behavior of the SSD method and how the choice of hyperparameters affect the decision boundary. We also try to address the shortcomings of SSD in two ways:

* By applying dampening at only the last layer of the model.
* By searching for hyperparameters whose dampening yields a loss as close as possible to the original model's loss on a "generalization dataset" constructed in the same way as the repair step for Scrub+R.

## Experiment runs
Now we run the experiment on each of the dataset configuration, moving the rogue point.

### 1. Rogue point with same distance to all centroids (close to decision boundary)
![Data configuration for Run 1](data/plots/data_1.png)

### Orignal model for the dataset
![Original model for Run 1](results/1_decision_boundary_retrain_original.png)

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="800"/></td>
    <td><img src="results/1_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/1_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/1_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/1_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/1_decision_boundary_ssd_v3_alpha=7.52_lambda=50.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>


</table>

### 2. Rogue point with same centroid as its class
![Data configuration for Run 2](data/plots/data_2.png)

### Orignal model for the dataset
![Original model for Run 2](results/2_decision_boundary_retrain_original.png)


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="800"/></td>
    <td><img src="results/2_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/2_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/2_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/2_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/2_decision_boundary_ssd_v3_alpha=5.37_lambda=14.29_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
<table>
### 3. Rogue point with same centroid as the class with a different label
![Data configuration for Run 3](data/plots/data_3.png)

### Orignal model for the dataset
![Original model for Run 3](results/3_decision_boundary_retrain_original.png)


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="800"/></td>
    <td><img src="results/3_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/3_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/3_decision_boundary_ssd_v3_alpha=6.34_lambda=15.67_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/3_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/3_decision_boundary_ssd_v3_alpha=6.34_lambda=15.67_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
<table>

### 4. Rogue point far away from its centroid, but probably in the same decision boundary
![Data configuration for Run 4](data/plots/data_4.png)

### Orignal model for the dataset
![Original model for Run 4](results/4_decision_boundary_retrain_original.png)


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="800"/></td>
    <td><img src="results/4_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/4_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/4_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/4_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/4_decision_boundary_ssd_v3_alpha=7.85_lambda=0.87_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
<table>



### 5. Rogue point far away from its centroid, but probably in the same decision boundary
![Data configuration for Run 5](data/plots/data_5.png)
<!-- TODO: Command to run experiment -->

### Orignal model for the dataset
![Original model for Run 5](results/5_decision_boundary_retrain_original.png)


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold">Unlearned</th>
    <th align="center" style="font-weight: bold">SSD</th>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="800"/></td>
    <td><img src="results/5_decision_boundary_ssd_alpha=1.00_lambda=1.00_unlearned.png" alt="ssd_org" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/5_decision_boundary_ssd_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/5_decision_boundary_ssd_v2_alpha=1.00_lambda=1.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/5_decision_boundary_ssd_v2_alpha=5.00_lambda=3.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Original Model" width="800"/></td>
    <td><img src="results/5_decision_boundary_ssd_v3_alpha=6.24_lambda=50.00_unlearned.png" alt="Retrained Model" width="800"/></td>
  </tr>
  <tr>
<table>