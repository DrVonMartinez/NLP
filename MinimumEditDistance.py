from typing import List, Tuple, Union, Any

import networkx as nx
import numpy as np
import pandas as pd


def min_edit_distance(word_x: str, word_y: str, levenshtein: bool = True) -> int:
    """
    Minimum number of editing operations needed to transform one into the other
    Operations include:
        Insertion
        Deletion
        Substitution
    :param word_x:
    :param word_y:
    :param levenshtein:
    :return:
    """
    insert_cost = 1 if levenshtein else 1
    delete_cost = 1 if levenshtein else 1
    substitute_cost = 2 if levenshtein else 1
    d = np.zeros((len(word_x) + 1, len(word_y) + 1), dtype=int)
    d[:, 0] = np.arange(len(word_x) + 1)
    d[0, :] = np.arange(len(word_y) + 1)
    ptr = np.zeros_like(d, dtype=str)
    ptr[0][0] = ' '
    if len(word_y) > 0:
        ptr[0][1:] = '<'
    if len(word_x) > 0:
        ptr[1:][0] = '^'
    for i in range(1, len(word_x) + 1):
        for j in range(1, len(word_y) + 1):
            insert = d[i - 1, j] + insert_cost
            delete = d[i, j - 1] + delete_cost
            substitute = d[i - 1, j - 1] + (substitute_cost if word_x[i - 1] != word_y[j - 1] else 0)
            d[i, j] = np.min([insert, delete, substitute])
            if d[i, j] == substitute:
                ptr[i][j] = '\\'
            elif d[i, j] == delete:
                ptr[i][j] = '<'
            else:
                ptr[i][j] = '^'
    char_swap_x = ''
    char_swap_y = ''
    i = len(word_x)
    j = len(word_y)
    while i > 0 or j > 0:
        if ptr[i][j] == '\\':
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
        else:
            raise ValueError(i, j, ptr[i, j])
    print('Word X: "' + char_swap_x + '"', 'Word Y: "' + char_swap_y + '"', sep='\n')
    return d[len(word_x), len(word_y)]


def weighted_min_edit_distance(word_x: str, word_y: str, weights: pd.DataFrame) -> int:
    """
    Minimum number of editing operations needed to transform one into the other
    Operations include:
        Insertion
        Deletion
        Substitution
    :param word_x:
    :param word_y:
    :param weights:
    :return:
    """
    d = np.zeros((len(word_x) + 1, len(word_y) + 1), dtype=int)
    for i in range(1, len(word_x) + 1):
        d[i, 0] = d[i - 1, 0] + weights['delete'][word_x[i - 1]]
    for j in range(1, len(word_y) + 1):
        d[0, j] = d[0, j - 1] + weights['insert'][word_y[j - 1]]

    ptr = np.zeros_like(d, dtype=str)
    ptr[0][0] = ' '
    if len(word_y) > 0:
        ptr[0][1:] = '<'
    if len(word_x) > 0:
        ptr[1:][0] = '^'
    for i in range(1, len(word_x) + 1):
        for j in range(1, len(word_y) + 1):
            delete = d[i - 1, j] + weights['delete'][word_x[i - 1]]
            insert = d[i, j - 1] + weights['insert'][word_y[j - 1]]
            substitute = d[i - 1, j - 1] + weights['substitute'][word_y[j - 1]]
            d[i, j] = np.min([insert, delete, substitute])
            if d[i, j] == substitute:
                ptr[i][j] = '\\'
            elif d[i, j] == insert:
                ptr[i][j] = '<'
            else:
                ptr[i][j] = '^'
    char_swap_x = ''
    char_swap_y = ''
    i = len(word_x)
    j = len(word_y)
    while i > 0 or j > 0:
        if ptr[i][j] == '\\':
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
        else:
            raise ValueError(i, j, ptr[i, j])
    print('Word X: "' + char_swap_x + '"', 'Word Y: "' + char_swap_y + '"', sep='\n')
    return d[len(word_x), len(word_y)]


def overlap_detection(word_x: str, word_y: str, local_threshold: int) -> list[tuple[Union[str, Any], Union[str, Any]]]:
    """
    Minimum number of editing operations needed to transform one into the other
    Operations include:
        Insertion
        Deletion
        Substitution
    :param word_x:
    :param word_y:
    :return:
    """
    insert_cost = 1
    delete_cost = 1
    substitute_cost = 2
    f = np.zeros((len(word_x) + 1, len(word_y) + 1), dtype=int)
    f[:, 0] = 0
    f[0, :] = 0
    ptr = np.zeros_like(f, dtype=str)
    ptr[0][0] = ''
    if len(word_y) > 0:
        ptr[0][1:] = ''
    if len(word_x) > 0:
        ptr[1:][0] = ''
    local_alignment: set = set()
    for i in range(1, len(word_x) + 1):
        for j in range(1, len(word_y) + 1):
            insert = f[i - 1, j] - 1
            delete = f[i, j - 1] - 1
            substitute = f[i - 1, j - 1] + (1 if word_x[i - 1] == word_y[j - 1] else -1)
            f[i, j] = np.max([0, insert, delete, substitute])
            if f[i, j] == substitute:
                ptr[i][j] = '\\'
            elif f[i, j] == delete:
                ptr[i][j] = '<'
            elif f[i, j] == insert:
                ptr[i][j] = '^'
            else:
                ptr[i][j] = ''
            if f[i, j] > local_threshold:
                local_alignment.add((i, j, f[i, j]))
    alignments_x = {}
    alignments_y = {}
    pathing = {}
    while len(local_alignment) > 0:
        i, j, _ = local_alignment.pop()
        char_swap_x = ''
        char_swap_y = ''
        max_found = -1
        path = []
        while f[i, j] > 0 and (i, j) not in local_alignment:
            if ptr[i][j] == '\\':
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
            else:
                raise ValueError(i, j, ptr[i, j])
            max_found = max([f[i, j], max_found])
            path += [(i, j)]
        alignments_x[char_swap_x] = max_found
        alignments_y[char_swap_y] = max_found
        pathing[char_swap_x] = {'path': path[::-1], 'visited': False, 'removed': False, 'index': len(alignments_x)}
    trim = sorted(pathing, key=lambda a: len(a), reverse=True)
    for i in trim:
        if pathing[i]['removed']:
            continue
        pathing[i]['visited'] = True
        others = list(filter(lambda x: x[-1] != i[-1], pathing))
        for other in others:
            o = pathing[other]['path']
            if pathing[other]['visited']:
                continue

            def is_contained_in():
                try:
                    sub_path = pathing[i]['path'][pathing[i]['path'].index(o[0]):]
                    print(pathing[i]['path'], sub_path, o)
                    return all([sub_path[q] == o[q] for q in range(len(sub_path))])
                except ValueError:
                    return False

            if is_contained_in():
                pathing[other]['removed'] = True
        pathing[i]['visited'] = True

    for path in pathing:
        if pathing[path]['removed']:
            alignments_x.pop(path)
            alignments_y.pop(path)
    return list(zip(alignments_x, alignments_y))


print(min_edit_distance('intention', 'execution'))
print(overlap_detection('ATTATC', 'ATCAT', 2))
