import pandas as pd

from Classification.nlp_classifier import f1_score


class NamedEntityRecognition:
    def __init__(self, class_names: list[str]):
        self.__categories = class_names
        self.__entities = []

    @staticmethod
    def tokenize(text: str):
        return text.split(' ')

    def evaluation(self, sample_passage: str, correct_entities: list[str]):
        rows = ['System Guess - Entity', 'System Guess - Other']
        columns = ['Object Property - Entity', 'Object Property - Other']
        confusion_matrix = pd.DataFrame([[0, 0], [0, 0]], index=rows, columns=columns)
        self.__entities = []
        tokens = self.tokenize(sample_passage)

        return f1_score(confusion_matrix)


if __name__ == '__main__':
    test1 = 'Foreign Ministry spokesman Shen Guofang told Reuters'
    io_encoding1 = ['Organization', 'Organization', 'Other', 'Person', 'Person', 'Other', 'Organization']
    iob_encoding1 = ['B-Organization', 'I-Organization', 'Other', 'B-Person', 'I-Person', 'Other', 'B-Organization']
    print(list(zip(test1.split(' '), io_encoding1)))
    print(list(zip(test1.split(' '), iob_encoding1)))
    test2 = "Fred showed Sue Mengqiu Huang 's new painting"
    io_encoding2 = ['Person', 'Other', 'Person', 'Person', 'Person', 'Other', 'Other', 'Other', 'Other']
    iob_encoding2 = ['B-Person', 'Other', 'B-Person', 'B-Person', 'I-Person', 'Other', 'Other', 'Other', 'Other']
    print(list(zip(test2.split(' '), io_encoding2)))
    print(list(zip(test2.split(' '), iob_encoding2)))
