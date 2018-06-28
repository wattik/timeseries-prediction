# coding=utf-8

class Scores:
    def __init__(self, scores):
        self.mean = scores.mean()
        self.std = scores.std()
        self.max = scores.max()
        self.min = scores.min()
        self.scores = scores

    def __str__(self):
        s = ""
        s += "mean: %.4f" % self.mean + "\n"
        s += "std: %.4f" % self.std + "\n"
        s += "min: %.4f" % self.min + "\n"
        s += "max: %.4f" % self.max + "\n"
        return s


class Results:
    def __init__(self, reference_scores: Scores, model_scores: Scores):
        self.model_scores = model_scores
        self.reference_scores = reference_scores

    def __str__(self):
        return "Reference: \n" + str(self.reference_scores) + "Model:\n" + str(self.model_scores)
