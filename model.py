from hmmlearn import hmm


class HMMModel:
    def __init__(self):
        # Use GaussianHMM model.
        # I changed n_components from 7 to 20 and at 10 accuracy was best.
        self.model = hmm.GaussianHMM(
            n_components=10,
            n_iter=1000
        )

    def shape(self):
        return [self.model.n_components, self.model.n_features]

    def train(self, x, lengths):
        self.model.fit(x, lengths)

    def evaluate(self, x, lengths=None):
        return self.model.score(x, lengths)
