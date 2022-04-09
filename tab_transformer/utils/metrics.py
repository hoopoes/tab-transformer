from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUC


def get_metric_collection() -> MetricCollection:
    metric_collection = MetricCollection([
        Accuracy(), Precision(), Recall(), AUC(reorder=True)
    ])

    return metric_collection
