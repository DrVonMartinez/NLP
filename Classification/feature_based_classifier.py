from functools import reduce

import numpy as np

from Classification.Features import f3_function, f2_function, f1_function


class Feature:
    def __init__(self, _class_: str, func):
        self.___class_: str = _class_
        self.___datum_ = func
        self.__weight: float = 1
        self.__name = str(func.__name__)

    def feature(self, args) -> bool:
        return self.___datum_(args)

    def get_class(self) -> str:
        return self.___class_

    def get_weight(self, args) -> float:
        return self.__weight * int(self.feature(args))

    def current_weight(self):
        return self.__weight

    def update_weight(self, weight: float):
        self.__weight = weight

    def __str__(self):
        return self.__name

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        name_check = str(other) == self.__name
        class_check = other.get_class() == self.___class_
        func_check = other.___datum_ == self.___datum_
        return all([name_check, class_check, func_check])


class ExponentialModels:
    def __init__(self, features=([], list[Feature])):
        self._classes: dict[str, list[int]] = {}
        self._features: list[Feature] = []
        for feature in features:
            _class_: str = feature.get_class()
            self.__add_class(class_name=_class_)
            self._classes[_class_].append(len(self._features))
            self._features.append(feature)

    def __add_class(self, class_name: str):
        self._classes.setdefault(class_name, [])

    def add_feature(self, feature: Feature):
        _class_ = feature.get_class()
        self.__add_class(class_name=_class_)
        self._classes[_class_].append(len(self._features))
        self._features.append(feature)

    def add_class(self, _class_: str):
        self.__add_class(_class_)

    def features(self) -> list:
        return [str(feature) for feature in self._features]

    def get_weights(self) -> list[tuple[str, float]]:
        return [(str(feature), feature.current_weight()) for feature in self._features]

    def vote(self, phrase: str):
        if not self._classes:
            raise ValueError("No classes are set")
        classes = list(self._classes)
        voting_value = [np.exp(sum([self._features[feature].get_weight(phrase) if feature != -1 else 0
                                    for feature in self._classes[class_name]]))
                        for class_name in self._classes]
        normalized = sum(voting_value)
        prob = {classes[i]: voting_value[i] / normalized for i in range(len(voting_value))}
        print(prob)
        assert sum(prob.values()) == 1
        return classes[np.argmax(list(prob.values()))]

    def actual_count(self, phrase: str, feature: Feature):
        index = self._features.index(feature)
        return sum([feature.feature(phrase) if index in self._classes[class_name] else 0
                    for class_name in self._classes])

    def predicted_count(self, phrase: str, feature: Feature):
        index = self._features.index(feature)
        denominator = sum([np.exp(sum([self._features[feature].get_weight(phrase)
                                       for feature in self._classes[class_name_ii]]))
                           for class_name_ii in self._classes])
        count = 0
        for C in self._classes:
            _numerator_ = 0
            for class_name_i in self._classes:
                numerator = np.exp(sum([self._features[feature].get_weight(phrase) for feature in self._classes[class_name_i]]))
                numerator2 = feature.feature(phrase) if index in self._classes[class_name_i] else 0
                _numerator_ += numerator * numerator2
            count += _numerator_ / denominator
        return count


def vote(phrase: str, features: list[Feature]):
    # list(map(lambda a: Feature.feature(a, phrase), feature_set))
    voting_index = []
    voting_value = []
    for feature in features:
        _class_ = feature.get_class()
        if _class_ not in voting_index:
            index = len(voting_index)
            voting_index.append(_class_)
            voting_value.append(0)
        else:
            index = voting_index.index(_class_)
        voting_value[index] += feature.get_weight(phrase)
    print(voting_value)
    max_value = np.argmax(voting_value)
    return voting_index[max_value]


if __name__ == '__main__':
    f1 = Feature('Location', f1_function)
    f4 = Feature('Location', f1_function)
    f1.update_weight(1.8)
    f2 = Feature('Location', f2_function)
    f2.update_weight(-0.6)
    f3 = Feature('Drug', f3_function)
    f3.update_weight(0.3)
    feature_set = [f1, f2, f3]
    sample_phrase = 'in Québec'
    sample_phrase2 = 'by Goéric'
    print(vote(sample_phrase, feature_set))
    exp_model_a = ExponentialModels(feature_set)
    exp_model_a.add_class('Person')
    print(exp_model_a.vote(sample_phrase))
    print(exp_model_a.vote(sample_phrase2))
    print(exp_model_a.features())
    print(f1 == f4)
    # Training form
    # log-linear, maxent, logistic, gibbs, svm, nn
