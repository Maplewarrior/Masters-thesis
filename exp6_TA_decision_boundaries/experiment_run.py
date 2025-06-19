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
    results_dir = os.path.join(os.path.dirname(__file__), "results", rogue)
    os.makedirs(results_dir, exist_ok=True)


    # ============== Load data ============== 
    dataloader_train, dataloader_val, dataloader_retain, dataloader_forget, forget_idxs = get_data(dataset_path, validation_path, cfg.data.batch_size, cfg.data.n_classes)
    
    dataset_name = cfg.data.dataset.split("/")[-1].split(".")[0]
    dataset_number = dataset_name.split("_")[1]

    # No logger, this could be changed to a wandb logger if needed
    logger = None 
    
    model = NeuralNet(M=dataloader_train.dataset.X.shape[1], n_classes=cfg.data.n_classes, seed=cfg.model.seed)

    original_model = copy.deepcopy(model)
    retrained_model = copy.deepcopy(model)

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
    trainer = NeuralNetworkTrainer(model=retrained_model, 
                                    train_dataloader=dataloader_retain, 
                                    val_dataloader=dataloader_val, 
                                    logger=logger, 
                                    device=cfg.model.device, 
                                    learning_rate=cfg.trainer.lr, 
                                    n_epochs=cfg.trainer.n_epochs, 
                                    disable_tqdm=cfg.trainer.disable_tqdm, 
                                    do_early_stopping=cfg.trainer.do_early_stopping)()

    plot_title = f"Retrained"
    decision_boundary_filename = f"{dataset_number}_decision_boundary_retrained"
    decision_boundary_plot(retrained_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

    from src.unlearners.teacher_ascend import TeacherAscender
    epochs = 20
    _lambda = 2 
    ta_model = copy.deepcopy(original_model)
    ta = TeacherAscender(ta_model, n_epochs=epochs, 
                        _lambda=_lambda, device=DEVICE)
    ta_metrics = ta(dataloader_retain, dataloader_forget, dataloader_val, eval=True, version="entropy-retain", is_FIM_ratio=True)
    
    plot_title = f"Teacher Ascend"
    decision_boundary_filename = f"{dataset_number}_decision_boundary_teacher_ascend"
    decision_boundary_plot(ta_model, original_model, dataloader_retain, dataloader_train, dataloader_forget.dataset.X, results_dir, plot_title=plot_title, filename=decision_boundary_filename)

if __name__ == "__main__":
    main()