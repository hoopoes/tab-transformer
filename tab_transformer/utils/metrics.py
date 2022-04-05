from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUC


def get_metric_collection() -> MetricCollection:
    metric_collection = MetricCollection([
        Accuracy(), Precision(), Recall()
    ])

    return metric_collection
