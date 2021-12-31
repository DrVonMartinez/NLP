import numpy as np
import pandas as pd
import enchant


word_list = enchant.Dict('en_US')
# assert False
# word_list = set(words.words())


# Creating a candidate set for word w:
#       Similar Pronunciations
#       Similar Spellings
#       Include word w

# Choosing candidate:
#       Noisy Channel
#       Classifier


def damerau_levenshtein_edit_distance(word_x: str, word_y: str, levenshtein: bool = True) -> int:
    """
    Minimum number of editing operations needed to transform one into the other
    Operations include:
        Insertion
        Deletion
        Substitution
        Transformation
    :param word_x:
    :param word_y:
    :param levenshtein:
    :return:
    """
    insert_cost = 1 if levenshtein else 1
    delete_cost = 1 if levenshtein else 1
    substitute_cost = 2 if levenshtein else 1
    transformation_cost = 1 if levenshtein else 1
    d = np.zeros((len(word_x) + 1, len(word_y) + 1), dtype=int)
    d[:, 0] = np.arange(len(word_x) + 1)
    d[0, :] = np.arange(len(word_y) + 1)
    ptr = np.zeros_like(d, dtype=str)
    ptr[0, 0] = ' '
    if len(word_y) > 0:
        ptr[0, 1:] = '<'
    if len(word_x) > 0:
        ptr[1:, 0] = '^'
    for i in range(1, len(word_x) + 1):
        for j in range(1, len(word_y) + 1):
            insert = d[i - 1, j] + insert_cost
            delete = d[i, j - 1] + delete_cost
            substitute = d[i - 1, j - 1] + (substitute_cost if word_x[i - 1] != word_y[j - 1] else 0)
            transform = d[i - 1, j - 1] + (transformation_cost
                                           if word_x[i - 2] != word_y[j - 1] or word_x[i - 1] != word_y[j - 2] else 0)
            d[i, j] = np.min([insert, delete, substitute, transform])
            if d[i, j] == substitute:
                ptr[i][j] = 'S'
            elif d[i, j] == transform:
                ptr[i][j] = 'T'
            elif d[i, j] == delete:
                ptr[i][j] = '<'
            else:
                ptr[i][j] = '^'
    char_swap_x = ''
    char_swap_y = ''
    i = len(word_x)
    j = len(word_y)
    while i > 0 or j > 0:
        if ptr[i][j] == 'S':
            char_swap_x = word_x[i - 1] + char_swap_x
            char_swap_y = '\x1B[3m' + word_y[j - 1] + '\x1B[0m' + char_swap_y
            i -= 1
            j -= 1
        elif ptr[i][j] == '<':
            char_swap_x = '*' + char_swap_x
            char_swap_y = word_y[j - 1] + char_swap_y
            j -= 1
        elif ptr[i][j] == '^':
            char_swap_x = word_x[i - 1] + char_swap_x
            char_swap_y = '*' + char_swap_y
            i -= 1
        elif ptr[i][j] == 'T':
            char_swap_x = word_x[i - 2] + word_x[i - 1] + char_swap_x
            char_swap_y = (word_y[j - 2] + "\u0332" + word_y[j - 1] + "\u0332") + char_swap_y
            i -= 2
            j -= 2
        else:
            raise ValueError(i, j, ptr[i, j])
    print(d)
    print(ptr)
    print('Word X: "' + char_swap_x + '"', 'Word Y: "' + char_swap_y + '"', sep='\n')
    return d[len(word_x), len(word_y)]


def spell_dist(word_x: str, word_y: str) -> int:
    """
    Minimum number of editing operations needed to transform one into the other
    Operations include:
        Insertion
        Deletion
        Substitution
        Transformation
    :param word_x:
    :param word_y:
    :return:
    """
    insert_cost = 1
    delete_cost = 1
    substitute_cost = 2
    transformation_cost = 1
    d = np.zeros((len(word_x) + 1, len(word_y) + 1), dtype=int)
    d[:, 0] = np.arange(len(word_x) + 1)
    d[0, :] = np.arange(len(word_y) + 1)
    for i in range(1, len(word_x) + 1):
        for j in range(1, len(word_y) + 1):
            insert = d[i - 1, j] + insert_cost
            delete = d[i, j - 1] + delete_cost
            substitute = d[i - 1, j - 1] + (substitute_cost if word_x[i - 1] != word_y[j - 1] else 0)
            transform = d[i - 1, j - 1] + (transformation_cost
                                           if word_x[i - 2] != word_y[j - 1] or word_x[i - 1] != word_y[j - 2] else 0)
            d[i, j] = np.min([insert, delete, substitute, transform])
    return d[len(word_x), len(word_y)]


if __name__ == '__main__':
    print(damerau_levenshtein_edit_distance('there', 'tehre'))
    # print(weighted_spell_dist('there', 'tehre'))


def one_step_spelling(error: str, misspell_rate=0.90):
    insert_cost = pd.read_csv('insert.csv', index_col=0).T
    delete_cost = pd.read_csv('delete.csv', index_col=0).T
    substitute_cost = pd.read_csv('substitution.csv', index_col=0).T
    reversal_cost = pd.read_csv('reversal.csv', index_col=0).T
    one_step: dict[str, float] = {}
    for c in range(len(error)):
        # insert
        test = error[:c] + error[c + 1:]
        if word_list.check(test):
            one_step[test] = insert_cost[error[c - 1]][error[c]]
        for i in 'abcdefghijklmnopqrstuvwxyz':
            # delete
            test = error[:c + 1] + i + error[c + 1:]
            if word_list.check(test):
                one_step[test] = delete_cost[error[c]][i]
            # substitute
            if i != c:
                test = error[:c] + i + error[c + 1:]
                if word_list.check(test):
                    one_step[test] = substitute_cost[i][error[c]]
        # Reversal
        if 1 < c < len(error):
            test = error[:c - 1] + error[c] + error[c - 1] + error[c + 1:]
            if word_list.check(test):
                one_step[test] = reversal_cost[error[c - 1]][error[c]]
    total = sum(one_step.values())
    one_step[error] = misspell_rate
    for key in one_step:
        if error != key:
            one_step[key] = (1 - misspell_rate) * one_step[key] / total
    return one_step


if __name__ == '__main__':
    print(one_step_spelling('acress'))
    print(one_step_spelling('two'))

