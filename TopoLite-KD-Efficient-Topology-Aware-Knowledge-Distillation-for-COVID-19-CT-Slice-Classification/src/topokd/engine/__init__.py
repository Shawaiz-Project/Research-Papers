"""Training and evaluation engines with lazy imports to avoid circular dependencies."""

__all__ = ["Trainer", "evaluate_splits", "predict_loader"]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    if name == "evaluate_splits":
        from .evaluator import evaluate_splits
        return evaluate_splits
    if name == "predict_loader":
        from .inference import predict_loader
        return predict_loader
    raise AttributeError(name)
