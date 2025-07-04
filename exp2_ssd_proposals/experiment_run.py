import hydra
import os
import copy
import matplotlib.pyplot as plt
from src.utils.get_git_root_path import get_git_root
from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.visualization.dampening_visualization import visualize_parameter_dampening
from src.plotting.decision_boundary_plot import decision_boundary_plot
from src.utils.load_rogue_data import get_data
from src.visualization.bo_plotting import plot_convergence_analysis
import numpy as np
import imageio

def count_updated_params(original_model, unlearned_model):
    pass

def ssd_bo_title(hyperparams):
    
    plt_tit = ""

    n_hyperparam_pairs = int((len(hyperparams["max"]) - 1)/2)
    for i in range(n_hyperparam_pairs):
        alpha = hyperparams["max"][f"alpha_{i}"]
        lamb = hyperparams["max"][f"_lambda_{i}"]
        num = i + 1
        plt_tit += f"$\\alpha_{num}={alpha:.2f}$, $\\lambda_{num}={lamb:.2f}$"
        if i < n_hyperparam_pairs - 1:  # Add separator except for last item
            if (i + 1) % 2 == 0:
                plt_tit += ",\n"
            else:
                plt_tit += ",   "
    plt_tit = plt_tit.rstrip(', ')

    return plt_tit

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    DEVICE = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')

    # ============= Preparing data and results directory =============
    git_root = get_git_root()
    dataset_path = cfg.data.dataset
    # path relative to current working directory
    dataset_path = os.path.join(git_root, dataset_path)
    validation_path = os.path.join("/",*dataset_path.split("/")[1:-1], "validation_data.npz")

    rogue = dataset_path.split("/")[-2] # rogue_many or rogue_one


    # ============== Load data ============== 
    dataloader_train, dataloader_val, dataloader_retain, dataloader_forget, forget_idxs = get_data(dataset_path, validation_path, cfg.data.batch_size, cfg.data.n_classes)
    
    dataset_name = cfg.data.dataset.split("/")[-1].split(".")[0]
    dataset_number = dataset_name.split("_")[1]

    # No logger, this could be changed to a wandb logger if needed
    logger = None 

    decision_boundary_filename = f"{dataset_number}_decision_boundary"
    
    dampening_plot_type = 'stacked'

    if cfg.unlearn.method == "retrain":

        model = NeuralNet(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)

        # Generate a simplified visualization
        nn_model_image_path = os.path.join(results_dir, f"nn_model_architecture")
        #create_simple_model_visualization(model, nn_model_image_path, X.shape[1])

        original_model = copy.deepcopy(model)

        # TODO Train a model on the retrain dataset, X_retrain, y_retrain
        print("Epochs: ", cfg.trainer.n_epochs)

        trainer = NeuralNetworkTrainer(model=original_model, 
                                       train_dataloader=dataloader_train, 
                                       val_dataloader=dataloader_val, 
                                       logger=logger, 
                                       device=cfg.model.device, 
                                       learning_rate=cfg.trainer.lr, 
                                       n_epochs=cfg.trainer.n_epochs, 
                                       disable_tqdm=cfg.trainer.disable_tqdm, 
                                       do_early_stopping=cfg.trainer.do_early_stopping)()
        
        # We train on retrain as it is the Retrain unlearning method
        trainer = NeuralNetworkTrainer(model=model, 
                                       train_dataloader=dataloader_retain, 
                                       val_dataloader=dataloader_val, 
                                       logger=logger, 
                                       device=cfg.model.device, 
                                       learning_rate=cfg.trainer.lr, 
                                       n_epochs=cfg.trainer.n_epochs, 
                                       disable_tqdm=cfg.trainer.disable_tqdm, 
                                       do_early_stopping=cfg.trainer.do_early_stopping)()

        results_dir = os.path.join(os.path.dirname(__file__), "results", cfg.unlearn.method, rogue)
        os.makedirs(results_dir, exist_ok=True)

       
        plot_title = f"Retrained"
        decision_boundary_plot(model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

    else:
        unlearned_model = NeuralNet(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
        NeuralNetworkTrainer(model=unlearned_model, 
                                    train_dataloader=dataloader_train, 
                                    val_dataloader=dataloader_val, 
                                    logger=logger, 
                                    device=cfg.model.device, 
                                    learning_rate=cfg.trainer.lr, 
                                    n_epochs=cfg.trainer.n_epochs, 
                                    disable_tqdm=cfg.trainer.disable_tqdm, 
                                    do_early_stopping=cfg.trainer.do_early_stopping)()

        original_model = copy.deepcopy(unlearned_model)
        
        if cfg.unlearn.method == "ssd":
            from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening
            # hyperparams = {'alpha': 30.,
            #                '_lambda': 5.}
            
            alpha = cfg.unlearn.ssd.alpha
            _lambda = cfg.unlearn.ssd._lambda

            ssd_model = SelectiveSynapticDampening(unlearned_model,
                                       alpha=alpha, 
                                       _lambda=_lambda,
                                       device=DEVICE)
            

            FIM_full = ssd_model.calculate_FIM(dataloader_train)
            FIM_forget = ssd_model.calculate_FIM(dataloader_forget)

            epsilon = 1e-12
            FIM_ratio = {key: FIM_forget[key] / (FIM_full[key] + epsilon) for key in FIM_full}


            dampenings = ssd_model(full_dataloader=dataloader_train,
                                   forget_dataloader=dataloader_forget,
                                   FIM_full=FIM_full,
                                   FIM_forget=FIM_forget,
                                   return_dampening=True)
                      
            
            # add hyperparameters to results folder name
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_alpha{alpha}_lambda{_lambda}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"SSD\n$\\alpha={alpha:.2f}$, $\\lambda={_lambda:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            

            fig = visualize_parameter_dampening(dampenings, type=dampening_plot_type)
            plt.savefig(f'{results_dir}/{dataset_number}_dampenings.pdf')

            plt.close()
            plt.clf()


            fig = visualize_parameter_dampening(FIM_ratio, type=dampening_plot_type, FIM_ratio=True, alpha_hyperparam=alpha)
            plt.savefig(f'{results_dir}/{dataset_number}_FIM_ratio_dampening_selection.pdf')

            fig = visualize_parameter_dampening(FIM_ratio, type=dampening_plot_type, FIM_ratio=True, alpha_hyperparam="no-value")
            plt.savefig(f'{results_dir}/{dataset_number}_FIM_ratio_dampening_selection_no_alpha.pdf')



            if cfg.unlearn.ssd.plot_gif:
                alpha_gif = 120
                gif_duration = 140
                gif_frames = 240
                alpha_linspace = np.linspace(0, alpha_gif, gif_frames)

                gif_dir = f'{results_dir}/gif'
                os.makedirs(gif_dir, exist_ok=True)
                image_paths = []
                for alpha in alpha_linspace:
                    fig = visualize_parameter_dampening(FIM_ratio, type=dampening_plot_type, FIM_ratio=True, alpha_hyperparam=alpha)
                    image_path = f'{gif_dir}/{dataset_number}_FIM_ratio_dampening_selection_alpha{alpha}.png'
                    plt.savefig(image_path)
                    image_paths.append(image_path)
                    plt.close()
                # Now make gif out of the images
                images = [imageio.imread(image_path) for image_path in image_paths]
                imageio.mimsave(f'{results_dir}/{dataset_number}_dampening_visualization.gif', images, duration=gif_duration, loop=0)


        elif cfg.unlearn.method == "ssd-layerwise":
            from src.unlearners.selective_synaptic_dampening_layerwise import SelectiveSynapticDampeningLayerwise as SelectiveSynapticDampening_ablation

            layer_target = cfg.unlearn.ssd_layerwise.layer_target
            alpha = cfg.unlearn.ssd_layerwise.alpha
            _lambda = cfg.unlearn.ssd_layerwise._lambda

            layer_target_actual = [(l - 1)*2 for l in layer_target] # ? Skip the relu between layers
            dampenings = SelectiveSynapticDampening_ablation(unlearned_model, 
                                            alpha=alpha, 
                                            _lambda=_lambda,
                                            layer_target=layer_target_actual,
                                            device=DEVICE)(full_dataloader=dataloader_train, 
                                                            forget_dataloader=dataloader_forget,
                                                            return_dampening=True)
            

            # Create structured directory hierarchy
            layer_target_str = '_'.join(map(str, layer_target)) if layer_target else 'all'
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_layers{layer_target_str}_alpha{alpha}_lambda{_lambda}", rogue)
            os.makedirs(results_dir, exist_ok=True)
            title = f'SSD\n${{L}}_{{target}}$ = {",".join(map(str, layer_target))}\n${{\\alpha}}$ = {alpha}, ${{\\lambda}}$ = {_lambda}'

            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=title, filename=decision_boundary_filename)
            
            # Create dampening visualization with structured path
            visualize_parameter_dampening(dampenings, type='stacked')
            # Save dampening plot in the same structured directory  
            dampening_filename = os.path.join(results_dir, f"{dataset_number}_dampenings.pdf")
            plt.savefig(dampening_filename)

        elif cfg.unlearn.method == "ssd-bo":
            from src.unlearners.selective_synaptic_dampening_BO import SelectiveSynapticDampeningBO
            k = cfg.unlearn.ssd_bo.k
            bo_plot_exploration_factor = cfg.unlearn.ssd_bo.bo_plot_exploration_factor

            hyperparams, dampenings = SelectiveSynapticDampeningBO(unlearned_model, 
                                                     n_bo_iter=100,
                                                     k=k,
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_k{k}", rogue)
            os.makedirs(results_dir, exist_ok=True)
            max_hyperparams = hyperparams['max']

            plot_title = f"BO-SSD\n$\\alpha={max_hyperparams['alpha']:.2f}$, $\\lambda={max_hyperparams['_lambda']:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings, type=dampening_plot_type)
            plt.savefig(f'{results_dir}/{dataset_number}_dampenings.pdf')

            # Plot convergence analysis
            plot_convergence_analysis(hyperparams['result'], 
                                    savepath=f'{results_dir}/{dataset_number}_convergence_{bo_plot_exploration_factor}.pdf',
                                    exploration_factor=bo_plot_exploration_factor)
            
        elif cfg.unlearn.method == "ssd-bo-smooth":
            from src.unlearners.selective_synaptic_dampening_BO_pairwise import SelectiveSynapticDampeningBOPairwise
            k = cfg.unlearn.ssd_bo_smooth.k
            P = cfg.unlearn.ssd_bo_smooth.P
            bo_plot_exploration_factor = cfg.unlearn.ssd_bo_smooth.bo_plot_exploration_factor

            hyperparams, dampenings = SelectiveSynapticDampeningBOPairwise(unlearned_model, 
                                                     P=P,
                                                     smooth_dampening=True,
                                                     n_bo_iter=100,
                                                     k=k,
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_P{P}_k{k}", rogue)

            os.makedirs(results_dir, exist_ok=True)
            max_hyperparams = hyperparams['max']

            plot_title = f"BO-SSD (smooth)\n$\\alpha={max_hyperparams['alpha_0']:.2f}$, $\\lambda={max_hyperparams['_lambda_0']:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings, type=dampening_plot_type)
            plt.savefig(f'{results_dir}/{dataset_number}_dampenings.pdf')

            # Plot convergence analysis
            plot_convergence_analysis(hyperparams['result'], 
                                    savepath=f'{results_dir}/{dataset_number}_convergence_{bo_plot_exploration_factor}.pdf',
                                    exploration_factor=bo_plot_exploration_factor)

        elif cfg.unlearn.method == "ssd-bo-pairwise":
            from src.unlearners.selective_synaptic_dampening_BO_pairwise import SelectiveSynapticDampeningBOPairwise
            P = cfg.unlearn.ssd_bo_pairwise.P
            k = cfg.unlearn.ssd_bo_pairwise.k

            bo_plot_exploration_factor = cfg.unlearn.ssd_bo_pairwise.bo_plot_exploration_factor

            hyperparams, dampenings = SelectiveSynapticDampeningBOPairwise(unlearned_model,
                                                     P=P,
                                                     k=k,
                                                     smooth_dampening=False,
                                                     n_bo_iter=100,
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_P{P}_k{k}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"BO-SSD\n{ssd_bo_title(hyperparams)}"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            fig = visualize_parameter_dampening(dampenings, type=dampening_plot_type)
            plt.savefig(f'{results_dir}/{dataset_number}_dampenings.pdf')


            # Plot convergence analysis
            plot_convergence_analysis(hyperparams['result'], 
                                    savepath=f'{results_dir}/{dataset_number}_convergence_{bo_plot_exploration_factor}.pdf',
                                    exploration_factor=bo_plot_exploration_factor)

        elif cfg.unlearn.method == "ssd-bo-pairwise-smooth":
            from src.unlearners.selective_synaptic_dampening_BO_pairwise import SelectiveSynapticDampeningBOPairwise
            P = cfg.unlearn.ssd_bo_pairwise_smooth.P
            k = cfg.unlearn.ssd_bo_pairwise_smooth.k

            bo_plot_exploration_factor = cfg.unlearn.ssd_bo_pairwise_smooth.bo_plot_exploration_factor

            hyperparams, dampenings = SelectiveSynapticDampeningBOPairwise(unlearned_model,
                                                     P=P,
                                                     k=k,
                                                     smooth_dampening=True,
                                                     n_bo_iter=100,
                                                     device=DEVICE)(full_dataloader=dataloader_train, 
                                                                    forget_dataloader=dataloader_forget,
                                                                    validation_dataloader=dataloader_val,
                                                                    return_dampening=True)
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_P{P}_k{k}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"BO-SSD (smooth)\n{ssd_bo_title(hyperparams)}"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings, type=dampening_plot_type)
            plt.savefig(f'{results_dir}/{dataset_number}_dampenings.pdf')

            # Plot convergence analysis
            plot_convergence_analysis(hyperparams['result'], 
                                    savepath=f'{results_dir}/{dataset_number}_convergence_{bo_plot_exploration_factor}.pdf',
                                    exploration_factor=bo_plot_exploration_factor)
    
        elif cfg.unlearn.method == "assd":
            from src.unlearners.adaptive_ssd import AdaptiveSSD
            hyperparams, dampenings = AdaptiveSSD(unlearned_model, device=DEVICE)(dataloader_train, dataloader_forget, return_dampening=True)


            alpha = hyperparams["alpha"]
            _lambda = hyperparams["_lambda"]
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"Adaptive SSD\n$\\alpha={alpha:.2f}$, $\\lambda={_lambda:.2f}$"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
            
            fig = visualize_parameter_dampening(dampenings, type=dampening_plot_type)
            plt.savefig(f'{results_dir}/{dataset_number}_dampenings.pdf')
        
        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()
