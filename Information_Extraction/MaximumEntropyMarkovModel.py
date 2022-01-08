from Information_Extraction.Features import Feature, has_digit
import nltk


class MaxEntMarkovModel:
    def __init__(self, features: list[Feature], classes: list[str]):
        self.__classes: set[str] = set([class_name for class_name in classes])
        self.__features: list[Feature] = self.__default_features() + features
        check_classes = set(map(lambda x: x.class_name, features))
        if any([class_name not in self.__classes for class_name in check_classes]):
            raise ValueError('Missing Classes')
        self.__context: tuple[str, str] = tuple()

    def __create_context(self, tokens: list[str]):
        self.__context = [('???', token) for token in tokens]

    def show_context(self):
        print(self.__context)

    @property
    def features(self):
        return [str(feature) for feature in self.__features]

    @features.setter
    def features(self, features: list[Feature]):
        raise PermissionError('Features are fixed by object creation')

    @property
    def weights(self):
        return [(str(feature), feature.weight) for feature in self.__features]


    def evaluate(self, tokens: list[str]):
        self.__create_context(tokens)
        for i in range(len(tokens)):
            for feature in self.__features:
                pass

    def __current_word(self, i: int):
        return self.__context[i][1]

    def __next_word(self, i: int):
        if i + 1 <= len(self.__context):
            return self.__context[i + 1][1]
        return None

    def __prior_word(self, i: int):
        if i > 0:
            return self.__context[i - 1][1]
        return None

    def __prior_context(self, i: int, history: int = 1):
        if i >= history:
            return '-'.join(map(lambda x: x[0], self.__context[i - history: i]))
        return None

    def __prior_bigram(self, i: int):
        if i >= 2:
            return '-'.join(map(lambda x: x[0], self.__context[i - 2: i]))
        return None

    def __default_features(self):
        results = []
        for class_name in self.__classes:
            results.append(Feature(class_name, self.__prior_context))
            results.append(Feature(class_name, self.__prior_bigram))
            results.append(Feature(class_name, self.__prior_word))
            results.append(Feature(class_name, self.__current_word))
            results.append(Feature(class_name, self.__next_word))
        return results


if __name__ == '__main__':
    with open('pos_tagging', 'r') as file:
        pos_classes = [line.removesuffix('\n') for line in file]
    print(pos_classes)
    local_features = [Feature(pos_class_name, has_digit) for pos_class_name in pos_classes]
    memm = MaxEntMarkovModel(local_features, pos_classes)
    print(memm.features, len(memm.features), sep='\n')
    print(memm.weights)


