import hydra
import os
import copy
from src.models.neural_network import NeuralNet
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.utils.get_git_root_path import get_git_root
from src.plotting.decision_boundary_plot import decision_boundary_plot
from src.utils.load_rogue_data import get_data

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

    if cfg.unlearn.method == "retrain":

        model = NeuralNet(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)

        original_model = copy.deepcopy(model)

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
        if cfg.unlearn.method not in ['amnesiac', 'sisa']:
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
        
        if cfg.unlearn.method == "scrubr":

            from src.unlearners.scrub import ScrubR

            alpha = 1
            gamma = 1
            n_rounds = 4
            n_repair_rounds = 6

            ScrubR(model=unlearned_model, 
                   original_model=original_model,
                   alpha=alpha,
                   gamma=gamma,
                   device=DEVICE)(retain_dataloader=dataloader_retain, 
                                  forget_dataloader=dataloader_forget, 
                                  val_dataloader=dataloader_val, 
                                  n_rounds=n_rounds, # min + max rounds
                                  n_repair_rounds=n_repair_rounds # min only rounds 
                                  )
            
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_alpha{alpha}_gamma{gamma}_n_rounds{n_rounds}_n_repair_rounds{n_repair_rounds}", rogue)
            os.makedirs(results_dir, exist_ok=True)
            plot_title = f"SCRUB+R"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
        
        elif cfg.unlearn.method == "ssd":
            from src.unlearners.selective_synaptic_dampening import SelectiveSynapticDampening

            alpha = 1
            _lambda = 1

            SelectiveSynapticDampening(unlearned_model, 
                                       alpha=alpha, 
                                       _lambda=_lambda,
                                       device=DEVICE)(full_dataloader=dataloader_train, 
                                                      forget_dataloader=dataloader_forget)
            
            # add hyperparameters to results folder name
            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_alpha{alpha}_lambda{_lambda}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"SSD"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
        
        elif cfg.unlearn.method == 'sae':
            from src.models.neural_network import NeuralNetRS
            from src.models.SAE import SAE
            from src.models.neural_net_with_sae import NeuralNetWithSAE
            from src.trainers.sae_trainer import SAETrainer
            from src.unlearners.sae_unlearner import SAEUnlearner

            alpha = 0.9
            layer_num = 6

            # instantiate & train neural network
            neural_net = NeuralNetRS(dataloader_train.dataset.X.shape[1], cfg.data.n_classes)
            nn_trainer = NeuralNetworkTrainer(neural_net, dataloader_train, dataloader_val)
            nn_trainer.train()
            original_model = copy.deepcopy(neural_net)

            # Make a retrained model
            neural_net_retrained = NeuralNetRS(dataloader_train.dataset.X.shape[1], cfg.data.n_classes)
            nn_trainer_retrained = NeuralNetworkTrainer(neural_net_retrained, dataloader_retain, dataloader_val)
            nn_trainer_retrained.train()

            # instantiate & train SAE
            sae = SAE(d=8, m=32, _lambda=1.).to(DEVICE)
            unlearned_model = NeuralNetWithSAE(neural_net, sae, layer_num=layer_num)
            sae_trainer = SAETrainer(unlearned_model, dataloader_train, dataloader_val, logger=None, device=DEVICE)            
            sae_trainer.train()

            # unlearn with feature dampening
            sae_unlearner = SAEUnlearner(unlearned_model, alpha=alpha, device=DEVICE)
            sae_unlearner(dataloader_retain, dataloader_forget)

            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_alpha{alpha}_layer_num{layer_num}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"SAE"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
      
            plot_title = f"Retrained (SAE)"
            retrained_decision_boundary_filename = f"{dataset_number}_retrained_decision_boundary"
            decision_boundary_plot(neural_net_retrained, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=retrained_decision_boundary_filename)

        elif cfg.unlearn.method == "amnesiac":
            from src.unlearners.amnesiac_unlearner import AmnesiacUnlearner
            from src.models.amnesiac_model import AmnesiacModel
            from src.trainers.amnesiac_trainer import AmnesiacTrainer
            
            unlearned_model = NeuralNet(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)
            # Wrap the unlearned model in an AmnesiacModel
            unlearned_model = AmnesiacModel(unlearned_model)

            # Train the unlearned model
            AmnesiacTrainer(model=unlearned_model, 
                            train_dataloader=dataloader_train, 
                            val_dataloader=dataloader_val, 
                            logger=logger, 
                            device=cfg.model.device,
                            learning_rate=cfg.trainer.lr,
                            weight_decay=cfg.trainer.weight_decay,
                            cache_gradients=True,
                            disable_tqdm=cfg.trainer.disable_tqdm,
                            do_early_stopping=cfg.trainer.do_early_stopping,
                            n_epochs=cfg.trainer.n_epochs,
                            save_dir=None)(indices_to_forget=forget_idxs)

            original_model = copy.deepcopy(unlearned_model)

            AmnesiacUnlearner(model=unlearned_model, 
                              unlearn_parameters=cfg.unlearn)(indices_to_forget=forget_idxs)

            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"Amnesiac"
            decision_boundary_plot(unlearned_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

        elif cfg.unlearn.method == "sisa":
            import shutil
            from src.unlearners.sisa_unlearner import SISAUnlearner
            from src.models.sisa_class import SISA
            from src.trainers.sisa_trainer import SISATrainer
            
            weights_dir = os.path.join(os.path.dirname(__file__), "weights")
            sisa_model_fn = NeuralNet
            sisa_model_parameters = {'M': dataloader_train.dataset.X.shape[1],
                                     'n_classes': cfg.data.n_classes,
                                     'seed': cfg.model.seed}
            sisa_optimzier_parameters = {'lr': cfg.trainer.lr}


            sisa = SISA(dataloader_train,
                        dataloader_val,
                        n_shards=cfg.sisa.n_shards, 
                        n_slices=cfg.sisa.n_slices,
                        model_fn=sisa_model_fn,
                        model_params=sisa_model_parameters,
                        optimizer_parms=sisa_optimzier_parameters,
                        n_classes=cfg.data.n_classes, 
                        n_epochs=cfg.trainer.n_epochs,
                        batch_size=cfg.data.batch_size,
                        save_dir=weights_dir+'/sisa',
                        device=DEVICE,
                        seed=cfg.model.seed) 
            
            SISATrainer(
                model=None, 
                sisa=sisa, 
                train_dataloader=dataloader_train, 
                val_dataloader=dataloader_val, 
                logger=None, 
                device=cfg.model.device, 
                learning_rate=cfg.trainer.lr, 
                n_epochs=cfg.trainer.n_epochs, 
                disable_tqdm=cfg.trainer.disable_tqdm, 
                do_early_stopping=cfg.trainer.do_early_stopping)()
        
            # original SISA model
            sisa_pre_unlearning = sisa.copy()

            # get unlearned SISA model
            sisa_unlearner = SISAUnlearner(sisa)
            sisa_unlearner(forget_indices=forget_idxs)

            results_dir = os.path.join(os.path.dirname(__file__), "results", f"{cfg.unlearn.method}_n_shards{cfg.sisa.n_shards}_n_slices{cfg.sisa.n_slices}", rogue)
            os.makedirs(results_dir, exist_ok=True)

            plot_title = f"SISA"
            decision_boundary_plot(sisa, sisa_pre_unlearning, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)
                                
            # remove saved SISA weights
            shutil.rmtree(f'{weights_dir}/sisa/{sisa.experiment_id}')

        else:
            raise NotImplementedError(f"Unlearning method {cfg.unlearn.method} not implemented")

if __name__ == "__main__":
    main()