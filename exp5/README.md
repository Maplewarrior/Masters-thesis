# Investigating Unlearning Techniques for Neural Networks

This repository contains the code for running experiments for SCRUB and Teacher Ascend (TA). The experiments are conducted on the MNIST dataset and are managed using Hydra for configuration.
## 1. Running Experiments

All experiments are managed through `exp5/experiment_run.py` and configured via `exp5/mnist_config.yaml`. Hydra is used for configuration management, which allows for easy command-line overrides.

### 1.1. Basic Experiment Run

To run a single experiment with the default parameters defined in `mnist_config.yaml`, execute the following command from the project's root directory:

```bash
python exp5/experiment_run.py
```

### 1.2. Overriding Configuration via Command Line

You can easily modify hyperparameters for a single run without editing the main configuration file.

Experiments run with the following command:

### Teacher Ascend Experiments Overview

| Lambda | Split Type | Split Details                     | Command                                                                                                                                                                                            | Status |
| :----- | :--------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| 64     | `tsne_box` | `x:[-4.2,-2.6], y:[-6.25,-4.8]`    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=64 data.split_type="tsne_box" data.tsne_box_coordinates.x_min=-4.2 data.tsne_box_coordinates.x_max=-2.6 data.tsne_box_coordinates.y_min=-6.25 data.tsne_box_coordinates.y_max=-4.8` | ✅          |
| 64     | `tsne_box` | `x:[-6.2,-3.4], y:[-1.6,1.0]`      | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=64 data.split_type="tsne_box" data.tsne_box_coordinates.x_min=-6.2 data.tsne_box_coordinates.x_max=-3.4 data.tsne_box_coordinates.y_min=-1.6 data.tsne_box_coordinates.y_max=1.0`  | ✅          |
| 64     | `tsne_box` | `x:[10.2,11.0], y:[-4.3,-0.4]`     | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=64 data.split_type="tsne_box" data.tsne_box_coordinates.x_min=10.2 data.tsne_box_coordinates.x_max=11.0 data.tsne_box_coordinates.y_min=-4.3 data.tsne_box_coordinates.y_max=-0.4` | ✅          |
| 64     | `random`   | `n_forget=50`                    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=64 data.split_type="random" data.n_forget_points=50`                                                              | ✅          |
| 64     | `random`   | `n_forget=100`                    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=64 data.split_type="random" data.n_forget_points=100`                                                              | ✅          |
| 64     | `random`   | `n_forget=200`                    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=64 data.split_type="random" data.n_forget_points=200`                                                              | ☐          |
| 2      | `tsne_box` | `x:[-4.2,-2.6], y:[-6.25,-4.8]`    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=2 data.split_type="tsne_box" data.tsne_box_coordinates.x_min=-4.2 data.tsne_box_coordinates.x_max=-2.6 data.tsne_box_coordinates.y_min=-6.25 data.tsne_box_coordinates.y_max=-4.8`  | ☐          |
| 2      | `tsne_box` | `x:[-6.2,-3.4], y:[-1.6,1.0]`      | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=2 data.split_type="tsne_box" data.tsne_box_coordinates.x_min=-6.2 data.tsne_box_coordinates.x_max=-3.4 data.tsne_box_coordinates.y_min=-1.6 data.tsne_box_coordinates.y_max=1.0`   | ☐          |
| 2      | `tsne_box` | `x:[10.2,11.0], y:[-4.3,-0.4]`     | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=2 data.split_type="tsne_box" data.tsne_box_coordinates.x_min=10.2 data.tsne_box_coordinates.x_max=11.0 data.tsne_box_coordinates.y_min=-4.3 data.tsne_box_coordinates.y_max=-0.4`  | ✅          |
| 2      | `random`   | `n_forget=50`                    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=2 data.split_type="random" data.n_forget_points=50`                                                               | ☐          |
| 2      | `random`   | `n_forget=100`                    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=2 data.split_type="random" data.n_forget_points=100`                                                               | ☐          |
| 2      | `random`   | `n_forget=200`                    | `python experiment_run.py unlearn.method=teacher_ascend unlearn.teacher_ascend._lambda=2 data.split_type="random" data.n_forget_points=200`                                                               | ☐          |

### Scrub Experiments

| Split Type | Split Details                     | Command                                                                                                                                                                          | Status |
| :--------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| `tsne_box` | `x:[-4.2,-2.6], y:[-6.25,-4.8]`    | `python experiment_run.py unlearn.method=scrub data.split_type="tsne_box" data.tsne_box_coordinates.x_min=-4.2 data.tsne_box_coordinates.x_max=-2.6 data.tsne_box_coordinates.y_min=-6.25 data.tsne_box_coordinates.y_max=-4.8` | ✅          |
| `tsne_box` | `x:[-6.2,-3.4], y:[-1.6,1.0]`      | `python experiment_run.py unlearn.method=scrub data.split_type="tsne_box" data.tsne_box_coordinates.x_min=-6.2 data.tsne_box_coordinates.x_max=-3.4 data.tsne_box_coordinates.y_min=-1.6 data.tsne_box_coordinates.y_max=1.0`  | ✅          |
| `tsne_box` | `x:[10.2,11.0], y:[-4.3,-0.4]`     | `python experiment_run.py unlearn.method=scrub data.split_type="tsne_box" data.tsne_box_coordinates.x_min=10.2 data.tsne_box_coordinates.x_max=11.0 data.tsne_box_coordinates.y_min=-4.3 data.tsne_box_coordinates.y_max=-0.4` | ✅          |
| `random`   | `n_forget=50`                    | `python experiment_run.py unlearn.method=scrub data.split_type="random" data.n_forget_points=50`                                                                                  | ✅          |
| `random`   | `n_forget=100`                    | `python experiment_run.py unlearn.method=scrub data.split_type="random" data.n_forget_points=100`                                                                                  | ✅          |
| `random`   | `n_forget=200`                    | `python experiment_run.py unlearn.method=scrub data.split_type="random" data.n_forget_points=200`                                                                                  | ☐          |