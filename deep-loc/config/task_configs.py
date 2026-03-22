"""
Task-specific configurations for easy switching between regression and classification.
"""

from config.default_config import DefaultConfig


class RegressionConfig(DefaultConfig):
    """Configuration for regression task (3D position estimation)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = 'regression'
        self.experiment_name = 'regression_baseline'


class ClassificationConfig(DefaultConfig):
    """Configuration for classification task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task = 'classification'
        self.experiment_name = 'classification_baseline'
        # Classification-specific parameters can be set here
        # self.num_classes will need to be determined from data


def get_config(task='regression', **kwargs):
    """
    Get configuration based on task.

    Args:
        task: 'regression' or 'classification'
        **kwargs: Additional parameters to override defaults

    Returns:
        Config object
    """
    if task == 'regression':
        return RegressionConfig(**kwargs)
    elif task == 'classification':
        return ClassificationConfig(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'regression' or 'classification'.")
