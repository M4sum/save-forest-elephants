from typing import Tuple

import torchmetrics

def get_metrics(device: str = 'cpu', num_labels=2) -> Tuple[dict, dict]:
    train_metrics = {
        'train_accuracy': torchmetrics.Accuracy(task='multilabel', num_labels=num_labels),
        'train_f1_score': torchmetrics.F1Score(task='multilabel', num_labels=num_labels),
        'train_precision': torchmetrics.Precision(task='multilabel', num_labels=num_labels),
        'train_recall': torchmetrics.Recall(task='multilabel', num_labels=num_labels),
    }

    val_metrics = {
        'val_accuracy': torchmetrics.Accuracy(task='multilabel', num_labels=num_labels),
        'val_f1_score': torchmetrics.F1Score(task='multilabel', num_labels=num_labels),
        'val_precision': torchmetrics.Precision(task='multilabel', num_labels=num_labels),
        'val_recall': torchmetrics.Recall(task='multilabel', num_labels=num_labels),
    }

    # metrics to cuda device
    for metric in train_metrics.values():
        metric.to(device)

    for metric in val_metrics.values():
        metric.to(device)
    return train_metrics, val_metrics
