import subprocess

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
methods =  ['ssd', 'assd', 'ssd_v6', 'ssd_v6_smooth', 'ssd_v7', 'scrubr', 'sae', 'amnesiac', 'sisa']
if __name__ == '__main__':
    import subprocess
    for seed in seeds:
        for unlearn_method in methods:
            subprocess.call([
                            "python3",
                            "exp4/experiment_run.py",
                            f"unlearn.method={unlearn_method}",
                            f"model.seed={seed}"
                            ])