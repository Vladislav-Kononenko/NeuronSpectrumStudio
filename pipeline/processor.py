from dataclasses import dataclass

@dataclass
class PipelineResult:
    timestamp: float
    features: object
    prediction: object | None

class CorrelationPipeline:
    def __init__(self, source, buffer, model=None):
        self.source = source
        self.buffer = buffer
        self.model = model

    def step(self):
        sample, timestamp = self.source.read_sample()
        self.buffer.append(sample, timestamp)

        if not self.buffer.is_ready():
            return None

        window, _ = self.buffer.get_all()

        from features.pearson import pearson_matrix
        from features.spearman import spearman_matrix
        from features.feature_vector import build_feature_vector

        p = pearson_matrix(window)
        s = spearman_matrix(window)
        features = build_feature_vector(p, s)

        prediction = None
        if self.model is not None:
            prediction = self.model.predict(features.reshape(1, -1))[0]

        return PipelineResult(
            timestamp=timestamp,
            features=features,
            prediction=prediction,
        )