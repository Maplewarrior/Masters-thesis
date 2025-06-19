#!/bin/bash

# Check if an experiment name was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 EXPERIMENT_NAME [HYDRA_ARG_1] [HYDRA_ARG_2] ..."
    exit 1
fi

# The first argument is the experiment name
BASE_EXPERIMENT_NAME="$1"
shift # Remove experiment name from argument list

# Append timestamp and random number to make experiment name unique
TIMESTAMP=$(date +%m%d_%H%M)
RANDOM_NUM=$((RANDOM % 1000))
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${TIMESTAMP}_${RANDOM_NUM}"

echo "Submitting job with experiment name: $EXPERIMENT_NAME"

# Properly escape all arguments to preserve quotes through shell processing
RUN_EXPERIMENTS_ARGS=""
for arg in "$@"; do
    # Use printf %q to properly escape the argument for shell safety
    escaped_arg=$(printf "%q" "$arg")
    RUN_EXPERIMENTS_ARGS="$RUN_EXPERIMENTS_ARGS $escaped_arg"
done
# Remove leading space
RUN_EXPERIMENTS_ARGS=${RUN_EXPERIMENTS_ARGS# }

# Source your environment variables
source ./env_vars.sh

# Path to the job script template
JOB_SCRIPT_TEMPLATE="./jobscript_template1.sh"

# Temporary job script file that will be populated with environment variables and the experiment name
TEMP_JOB_SCRIPT="./jobscript_populated.sh"

# Replace placeholders in the template.
# NOTE: We use a different delimiter for sed (#) because RUN_EXPERIMENTS_ARGS can contain slashes or other special characters.
sed -e "s#\${VENV_PATH}#$VENV_PATH#g" \
    -e "s#\${WORKING_DIR}#$WORKING_DIR#g" \
    -e "s#\${EMAIL}#$EMAIL#g" \
    -e "s#\${QUEUE}#$QUEUE#g" \
    -e "s#\${WALLTIME}#$WALLTIME#g" \
    -e "s#\${GPU_MODE}#$GPU_MODE#g" \
    -e "s#\${NUM_CORES}#$NUM_CORES#g" \
    -e "s#\${MEM_GB}#$MEM_GB#g" \
    -e "s#\${EXPERIMENT_NAME}#$EXPERIMENT_NAME#g" \
    -e "s#\${RUN_EXPERIMENTS_ARGS}#$RUN_EXPERIMENTS_ARGS#g" \
    "$JOB_SCRIPT_TEMPLATE" > "$TEMP_JOB_SCRIPT"

# Submit the job
bsub < "$TEMP_JOB_SCRIPT"

# Optionally, remove the temporary job script after submission
rm "$TEMP_JOB_SCRIPT"
