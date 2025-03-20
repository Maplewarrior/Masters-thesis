# Experiment 1: Rogue One :octocat:
In this experiment we wish to explore what happens with the decision boundary depending on where the point to forget is positioned (for the synthetic data we call this the "rouge point"). We experiment with placing it in a wrong class, 

> [!NOTE]  
> The Amnesiac unlearning method does not have a repair step. Might not be necessary for this experiment as we are forgetting a single point and points are easily separable.

> [!NOTE]  
> The SAE model is slightly different from the other models (it uses skip connections) and the results before unlearning will be different than the others, even when using the same seed.
> The same is the case for the SISA model as it is an ensemble of models.

## How to use
To generate the data, run:
```bash
python exp1_rogue_one/generate_data.py
```

To run the experiment for all the datasets and unlearning methods, run:
```bash
python exp1_rogue_one/experiment_run.py data.dataset="data/data_1.npz,data/data_2.npz,data/data_3.npz,data/data_4.npz,data/data_5.npz" unlearn.method="amnesiac,ssd,retrain,scrubr,sae,sisa"  --multirun
```

You can overwrite any parameters from the config file, for example `n_epochs`:
```bash
python exp1_rogue_one/experiment_run.py trainer.n_epochs=10
```


## The network
The network is a simple feedforward network which can be seen below:
![Network architecture](results/nn_model_architecture.png)

## Experiment runs
Now we run the experiment on each of the dataset configuration, moving the rogue point.

### 1. Rogue point with same distance to all centroids (close to decision boundary)
![Data configuration for Run 1](data/plots/data_1.png)

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold"></th>
    <th align="center" style="font-weight: bold">Before unlearning</th>
    <th align="center" style="font-weight: bold">After unlearning</th>
  </tr>
  <tr>
    <td>Retrain</td>
    <td><img src="results/1_decision_boundary_retrain_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/1_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Scrub+R</td>
    <td><img src="results/1_decision_boundary_scrubr_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/1_decision_boundary_scrubr_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SSD</td>
    <td><img src="results/1_decision_boundary_ssd_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/1_decision_boundary_ssd_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Amnesiac</td>
    <td><img src="results/1_decision_boundary_amnesiac_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/1_decision_boundary_amnesiac_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SAE</td>
    <td><img src="results/1_decision_boundary_sae_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/1_decision_boundary_sae_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SISA</td>
    <td><img src="results/1_decision_boundary_sisa_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/1_decision_boundary_sisa_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
</table>

### 2. Rogue point with same centroid as its class
![Data configuration for Run 2](data/plots/data_2.png)

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold"></th>
    <th align="center" style="font-weight: bold">Before unlearning</th>
    <th align="center" style="font-weight: bold">After unlearning</th>
  </tr>
  <tr>
    <td>Retrain</td>
    <td><img src="results/2_decision_boundary_retrain_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/2_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Scrub+R</td>
    <td><img src="results/2_decision_boundary_scrubr_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/2_decision_boundary_scrubr_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
    <tr>
    <td>SSD</td>
    <td><img src="results/2_decision_boundary_ssd_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/2_decision_boundary_ssd_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Amnesiac</td>
    <td><img src="results/2_decision_boundary_amnesiac_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/2_decision_boundary_amnesiac_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SAE</td>
    <td><img src="results/2_decision_boundary_sae_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/2_decision_boundary_sae_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SISA</td>
    <td><img src="results/2_decision_boundary_sisa_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/2_decision_boundary_sisa_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
</table>


### 3. Rogue point with same centroid as the class with a different label
![Data configuration for Run 3](data/plots/data_3.png)

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold"></th>
    <th align="center" style="font-weight: bold">Before unlearning</th>
    <th align="center" style="font-weight: bold">After unlearning</th>
  </tr>
  <tr>
    <td>Retrain</td>
    <td><img src="results/3_decision_boundary_retrain_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/3_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Scrub+R</td>
    <td><img src="results/3_decision_boundary_scrubr_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/3_decision_boundary_scrubr_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SSD</td>
    <td><img src="results/3_decision_boundary_ssd_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/3_decision_boundary_ssd_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Amnesiac</td>
    <td><img src="results/3_decision_boundary_amnesiac_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/3_decision_boundary_amnesiac_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SAE</td>
    <td><img src="results/3_decision_boundary_sae_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/3_decision_boundary_sae_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SISA</td>
    <td><img src="results/3_decision_boundary_sisa_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/3_decision_boundary_sisa_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
</table>

### 4. Rogue point far away from its centroid, but probably in the same decision boundary
![Data configuration for Run 4](data/plots/data_4.png)

#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold"></th>
    <th align="center" style="font-weight: bold">Before unlearning</th>
    <th align="center" style="font-weight: bold">After unlearning</th>
  </tr>
  <tr>
    <td>Retrain</td>
    <td><img src="results/4_decision_boundary_retrain_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/4_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Scrub+R</td>
    <td><img src="results/4_decision_boundary_scrubr_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/4_decision_boundary_scrubr_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SSD</td>
    <td><img src="results/4_decision_boundary_ssd_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/4_decision_boundary_ssd_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Amnesiac</td>
    <td><img src="results/4_decision_boundary_amnesiac_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/4_decision_boundary_amnesiac_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SAE</td>
    <td><img src="results/4_decision_boundary_sae_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/4_decision_boundary_sae_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SISA</td>
    <td><img src="results/4_decision_boundary_sisa_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/4_decision_boundary_sisa_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
</table>



### 5. Rogue point far away from its centroid, but probably in the same decision boundary
![Data configuration for Run 5](data/plots/data_5.png)
<!-- TODO: Command to run experiment -->


#### Findings

<table>
  <tr>
    <th align="center" style="font-weight: bold"></th>
    <th align="center" style="font-weight: bold">Before unlearning</th>
    <th align="center" style="font-weight: bold">After unlearning</th>
  </tr>
  <tr>
    <td>Retrain</td>
    <td><img src="results/5_decision_boundary_retrain_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/5_decision_boundary_retrain_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Scrub+R</td>
    <td><img src="results/5_decision_boundary_scrubr_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/5_decision_boundary_scrubr_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SSD</td>
    <td><img src="results/5_decision_boundary_ssd_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/5_decision_boundary_ssd_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>Amnesiac</td>
    <td><img src="results/5_decision_boundary_amnesiac_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/5_decision_boundary_amnesiac_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SAE</td>
    <td><img src="results/5_decision_boundary_sae_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/5_decision_boundary_sae_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
  <tr>
    <td>SISA</td>
    <td><img src="results/5_decision_boundary_sisa_original.png" alt="Original Model" width="400"/></td>
    <td><img src="results/5_decision_boundary_sisa_unlearned.png" alt="Retrained Model" width="400"/></td>
  </tr>
</table>