import numpy as np

from Classification.Features import f1_function, f2_function, f3_function, is_capital
from feature_based_classifier import ExponentialModels, Feature


class MaxEntropy(ExponentialModels):
    def __init__(self, features=([], list[Feature])):
        super(MaxEntropy, self).__init__(features)
        self.__gradient = [[]] * len(self._features)

    def log_likelihood(self, phrase: str):
        voting_value = np.array([np.exp(sum([self._features[feature].get_weight(phrase) if feature != -1 else 0
                                             for feature in self._classes[class_name]]))
                                 for class_name in self._classes])
        normalized = sum(voting_value)
        return voting_value / normalized

    def derivative(self, phrase: str, feature: Feature):
        return self.actual_count(phrase, feature) - self.predicted_count(phrase, feature)

    def train(self, phrases: list[str], training_rate=0.01):
        """
        Quasi-Newton method, L-BFGS
        :param phrases:
        :param training_rate:
        :return:
        """
        # self.__gradient = np.array([0] * len(self._features))
        for feature_index in range(len(self._features)):
            self.__gradient[feature_index] = []
            for phrase in phrases:
                self.__gradient[feature_index].append(self.derivative(phrase, self._features[feature_index]))
        print('Gradient', self.__gradient)
        for feature_index in range(len(self._features)):
            current_weight = self._features[feature_index].current_weight()
            new_weight = current_weight + np.average(self.__gradient[feature_index]) * training_rate
            self._features[feature_index].update_weight(new_weight)
            # print(phrase, self.get_weights())
        return self.get_weights()


if __name__ == '__main__':
    f1 = Feature('Location', f1_function)
    f4 = Feature('Location', f1_function)
    f1.update_weight(1.8)
    f2 = Feature('Location', f2_function)
    f2.update_weight(-0.6)
    f3 = Feature('Drug', f3_function)
    f3.update_weight(0.3)
    f4 = Feature('Location', is_capital)
    feature_set = [f1, f2, f3, f4]
    sample_phrase = 'in Québec'
    sample_phrase2 = 'by Goéric'
    sample_phrase3 = 'in London'
    sample_phrase4 = 'by Human'
    maxent_a = MaxEntropy(feature_set)
    maxent_a.add_class('Person')
    for i in range(10000):
        print(maxent_a.train([sample_phrase, sample_phrase2, sample_phrase3]))
    # Training form
    # log-linear, maxent, logistic, gibbs, svm, nn
