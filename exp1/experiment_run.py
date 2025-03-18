
import hydra
import wandb
from src.modelling.UnlearningManager import UnlearningManager

def load_dataset():
    # TODO load dataset
    pass

@hydra.main(config_path="configs/exp1", config_name="config")
def main(cfg):
    print(cfg)

    model = cfg.model.name
    n_epochs = cfg.model.n_epochs
    lr = cfg.model.lr
    batch_size = cfg.model.batch_size
    seed = cfg.model.seed


    logger = wandb.init(project_name="exp1", config=cfg)

    unlearning_manager = UnlearningManager(config=cfg, device="cuda", wandb=logger, track_performance=True)
    data = load_dataset("")
    
    if model.name == "retrain":
        raise NotImplementedError("Retrain is not implemented")

        data.retain
        # TODO train model on retrain dataset
        
    elif model.name == "unlearn1":
        raise NotImplementedError("Retrain is not implemented")
        pass
    elif model.name == "unlearn2":
        pass


if __name__ == "__main__":
    main()
