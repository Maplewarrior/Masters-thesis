# Experiment 1
In this experiment we wish to explore what happens with the decision boundary depending on where the point to forget is positioned (for the synthetic data we call this the "rouge point"). We experiment with placing it in a wrong class, 


## Data generation
To generate the data, run:
```bash
python exp1/generate_data.py
```

## The network

## Experiment runs
Now we run the experiment on each of the dataset configuration, moving the rogue point.

### 1. Rogue point with same distance to all centroids (close to decision boundary)
![Data configuration for Run 1](data/plots/data_1.png)

#### Expectation
We expect that unlearning this point will confuse the model and decision boundary will change more than for the other setups.

#### Run the experiment

Run the experiment
```bash
python exp1/experiment_run.py
```

#### Findings
<!-- TODO: Document findings after running experiment -->


### 2. Rogue point with same centroid as its class
![Data configuration for Run 2](data/plots/data_2.png)

#### Expectation
<!-- TODO: What do we expect -->


#### Run the experiment
<!-- TODO: Command to run experiment -->


#### Findings
<!-- TODO: Document findings after running experiment -->



### 3. Rogue point with same centroid as the class with a different label
![Data configuration for Run 3](data/plots/data_3.png)

#### Expectation
<!-- TODO: What do we expect -->


#### Run the experiment
<!-- TODO: Command to run experiment -->


#### Findings
<!-- TODO: Document findings after running experiment -->


### 4. Rogue point far away from its centroid, but probably in the same decision boundary
![Data configuration for Run 4](data/plots/data_4.png)

#### Expectation
<!-- TODO: What do we expect -->


#### Run the experiment
<!-- TODO: Command to run experiment -->


#### Findings
<!-- TODO: Document findings after running experiment -->




### 5. Rogue point far away from its centroid, but probably in the same decision boundary
![Data configuration for Run 5](data/plots/data_5.png)

#### Expectation
<!-- TODO: What do we expect -->


#### Run the experiment
<!-- TODO: Command to run experiment -->


#### Findings
<!-- TODO: Document findings after running experiment -->
