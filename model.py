from hmmlearn import hmm


class HMMModel:
    def __init__(self):
        self.model = hmm.GaussianHMM(
            n_components=10,
            n_iter=1000
        )

    def shape(self):
        return [self.model.n_components, self.model.n_features]

    def train(self, x):
        self.model.fit(x)

    def evaluate(self, x):
        return self.model.score(x)
