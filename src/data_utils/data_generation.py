from src.data_utils.synthetic_data import DataGenerator
from src.utils.config_loader import Config
from src.data_utils.synthetic_data import create_dataloaders

def generate_data(args) -> DataGenerator:
    """
    Generates and splits the synthetic data using arguments from argparse.
    
    Args:
        args: Parsed command line arguments containing data generation parameters
    """
    data_generator = DataGenerator(random_state=args.random_state)
    data_generator.generate_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        n_outliers=args.n_outliers,
        outlier_scale=args.outlier_scale,
        outlier_variance=args.outlier_variance,
        outlier_class=args.outlier_class
    )
    data_generator.split_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    return data_generator


def create_forget_retain_split(data_generator, config: Config):
    """Creates a new forget/retain split for existing data and returns associated dataloaders."""
    data_generator.draw_forget_set(
        n_points=config.forget.n_points, 
        class_idx=config.forget.class_idx, 
        ood_ratio=config.forget.ood_ratio
    )
    
    # For amnesiac, we need indices, so batch_size differs:
    if config.experiment.unlearn_type == "amnesiac":
        dataloaders = create_dataloaders(
            data_generator, 
            batch_size=32, 
            use_indices=True, 
            device=config.system.device
        )
    else:
        dataloaders = create_dataloaders(
            data_generator, 
            batch_size=32, 
            onehot_labels=True, 
            device=config.system.device
        )
    
    return dataloaders
