import numpy as np
from nltk.corpus import reuters
import pandas as pd
from functools import reduce

word_set = reuters.words()
# doc_set = reuters.
print(reuters.words(), len(word_set))

if __name__ == '__main__':
    data = [[95, 1, 13, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [10, 90, 0, 1, 0, 0],
            [0, 0, 0, 34, 3, 7],
            [np.nan, 1, 2, 13, 26, 5],
            [0, 0, 2, 14, 5, 10]]
    topics = ['UK', 'poultry', 'wheat', 'coffee', 'interest', 'trade']
    rows = ['True ' + topic for topic in topics]
    columns = ['Assigned ' + topic for topic in topics]
    reuters_sample = pd.DataFrame(data=data, columns=columns, index=rows)
    print(reuters_sample)


def split_confusion_matrix(confusion_matrix: pd.DataFrame, base_class_names: list[str]) -> list[pd.DataFrame]:
    _rows = ['Classifier: yes', 'Classifier: no']
    _columns = ['Truth: yes', 'Truth: no']
    stack = []
    for c in base_class_names:
        mini_confusion = pd.DataFrame(index=_rows, columns=_columns)
        true_true = confusion_matrix['Assigned ' + c]['True ' + c]
        true_false = confusion_matrix.loc['True ' + c].sum() - true_true
        false_true = confusion_matrix['Assigned ' + c].sum() - true_true
        false_false = confusion_matrix.to_numpy(na_value=0).sum() - true_false - false_true + true_true
        mini_confusion[_columns[0]][_rows[0]] = true_true
        mini_confusion[_columns[0]][_rows[1]] = true_false
        mini_confusion[_columns[1]][_rows[0]] = false_true
        mini_confusion[_columns[1]][_rows[1]] = false_false
        stack.append(mini_confusion)
    return stack


def precision(dataset: pd.DataFrame) -> float:
    return dataset['Classifier: yes']['Truth: yes'] / dataset['Classifier: yes'].sum()


def recall(dataset: pd.DataFrame) -> float:
    return dataset['Classifier: yes']['Truth: yes'] / dataset.loc['Truth: yes'].sum()


def accuracy(dataset: pd.DataFrame) -> float:
    return (dataset['Classifier: yes']['Truth: yes'] + dataset['Classifier: no']['Truth: no']) \
           / dataset.to_numpy().sum()


def f1_score(dataset: pd.DataFrame) -> float:
    _recall_ = recall(dataset)
    _precision_ = precision(dataset)
    return 2 * (_precision_ * _recall_) / (_precision_ + _recall_)



def macro_averaging(func, dataset: list[pd.DataFrame]):
    """
    Macro-averaging:
       Compute performance for each class, then average
    :param func:
    :param dataset:
    :return:
    """
    return np.average(list(map(func, dataset)))


def micro_averaging(func, dataset: list[pd.DataFrame]):
    """
    Micro-averaging:
    Collect decisions for all classes
        Compute Contingency Table -> Evaluate
    :param func:
    :param dataset:
    :return:
    """
    return func(reduce(lambda x, y: x.add(y), dataset))


if __name__ == '__main__':
    # macro_averaging(reuters_sample, topics)
    columns = ['Classifier: yes', 'Classifier: no']
    rows = ['Truth: yes', 'Truth: no']
    class_1 = pd.DataFrame(data=[[10, 10], [10, 970]], index=rows, columns=columns)
    class_2 = pd.DataFrame(data=[[90, 10], [10, 890]], index=rows, columns=columns)
    result = [class_1, class_2]
    print('Precision 1:', precision(class_1), '\nPrecision2:', precision(class_2))
    print('Macro-Average Precision:', macro_averaging(precision, result))
    print('Micro-Average Precision:', micro_averaging(precision, result))


def binarized_naive_bayes(document: str):
    binary_check = set([i for i in document])

