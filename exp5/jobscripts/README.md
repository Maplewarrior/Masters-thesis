This guide and jobscript template is adapted from the [HowLowCanYouGo repository](https://github.com/AndreasLF/HowLowCanYouGo/tree/master/GraphEmbeddings/jobscripts) repository.

# Running experiments on HPC cluster
We are using the HPC cluster at the Techincal University of Denmark to run our experiments. More info can be found [here](https://www.hpc.dtu.dk/).

## Accessing the cluster
To access the cluster you need to have a DTU account and be on the DTU network or VPN. SSH into the cluster using the following command:

```sh
ssh login.gbar.dtu.dk
```


## Jobscripts
For reproducibility purpososes we have provided a jobscript template along with a submit_job script. The experiment config files can all be found in the [configs folder](https://github.com/Maplewarrior/Masters-thesis/blob/main/exp5/mnist_config.yaml).

To be able to run the jobscripts you need to create an `env_vars.sh` and place it in the [jobscrips folder](https://github.com/Maplewarrior/Masters-thesis/blob/main/exp5/jobscipts). Below is an example of the contents of an `env_vars.sh` file:

```sh
export WORKING_DIR=<working_directory>
export VENV_PATH="$WORKING_DIR/venv/bin/activate"
export EMAIL=<email>
export QUEUE="gpua100"
export NUM_CORES=4
export GPU_MODE="num=1:mode=exclusive_process"
export WALLTIME="12:00"
export MEM_GB=16
```

## Submitting a job

### Jobscript 1 (Find lowest rank representation of graph)
To submit a job, run the following command in the terminal:

```sh
./submit_job1.sh EXPERIMENT_ARGS
```

Where `EXPERIMENT_ARGS` is the arguments for overwriting the Hydra experiment arguments.

> [!NOTE]  
> Make sure to escape any inner quotes in the `EXPERIMENT_ARGS` string. E.g. `unlearn.methods.teacher_ascend.versions='[["ce", false], ["entropy", false]]'` should be `unlearn.methods.teacher_ascend.versions='[[\"ce\", false], [\"entropy\", false]]'`

This will populate the `jobscript_template1.sh` with the parameters defined in `env_vars.sh` and submit the job to the cluster.
