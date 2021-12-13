import numpy as np
from nltk.corpus import words

word_list = set(words.words())
print(len(word_list))


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
            char_swap_y = word_y[j - 1] + char_swap_y
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
    :param levenshtein:
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


print(damerau_levenshtein_edit_distance('there', 'tehre'))


def one_step_spelling(noisy_word: str):
    one_step: set = set()
    # delete
    for c in range(len(noisy_word)):
        test = noisy_word[:c] + noisy_word[c + 1:]
        if test in word_list:
            one_step.add(test)
        for i in 'abcdefghijklmnopqrstuvwxyz -':
            # insert
            test = noisy_word[:c + 1] + i + noisy_word[c + 1:]
            if test in word_list:
                one_step.add(test)
            # substitute
            if i != c:
                test = noisy_word[:c] + i + noisy_word[c + 1:]
                if test in word_list:
                    one_step.add(test)
        if 0 < c < len(noisy_word):
            test = noisy_word[:c - 1] + noisy_word[c] + noisy_word[c - 1] + noisy_word[c + 1:]
            if test in word_list:
                one_step.add(test)
    return sorted(one_step)


print(one_step_spelling('acress'))


def noisy_channel(noisy_word: str) -> str:
    """
    w_ = argmax P(x|w) P(w)
    P(x|w): Channel Model / Error Model
    P(w): Language Model
    :param noisy_word
    :return
    """
