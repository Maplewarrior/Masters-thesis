import subprocess

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

mnist_methods =  ['ssd', 'assd', 'ssd_v6', 'ssd_v6_smooth', 'ssd_v6_paired', 'scrubr', 'sae', 'amnesiac', 'sisa', 'teacher-ascend']
cifar_methods =  ['ssd', 'assd', 'ssd_v6', 'ssd_v6_smooth', 'ssd_v6_paired', 'scrubr', 'sae', 'teacher-ascend'] 

if __name__ == '__main__':
    import subprocess
    for seed in seeds:
        for unlearn_method in cifar_methods:
            subprocess.call([
                            "python3",
                            "exp4/experiment_run.py",
                            f"unlearn.method={unlearn_method}",
                            f"model.seed={seed}"
                            ])