# Base config file
BASE_CONFIG="configs/config.yaml"

# Models to test
MODELS=("ssd" "scrub+r")

# Run sweeps for different parameters
echo "Running sweep for number of samples..."
python src/experiment/parameter_sweep.py \
    --base_config $BASE_CONFIG \
    --param forget.ood_ratio \
    --min 0.0 \
    --max 1.0 \
    --steps 10 \
    --output_dir "configs/sweep/forget_ood_ratio" \
    --models ${MODELS[@]}
# echo "Running sweep for outlier scale..."
# python src/experiment/parameter_sweep.py --base_config $BASE_CONFIG --param data.outlier_scale --min 1.0 --max 10.0 --steps 5 --log_scale

# echo "Running sweep for forget points..."
# python src/experiment/parameter_sweep.py --base_config $BASE_CONFIG --param forget.n_points --min 10 --max 100 --steps 10

echo "All parameter sweeps completed!" 