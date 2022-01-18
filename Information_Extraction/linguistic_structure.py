import nltk


# Constituency (Phrase Structure)
#       Distribution:
#           A constituent behaves as a unit that can appear in different places
#       Substitution/Expansion/Pro-forms:
#       Coordination
#       Regular Internal Structure
#       No intrusion
#       Fragments
#       Semantics
#       etc
# Dependency
#         ___
# The boy put the tortoise on the rug
# The ^^^ --- the ^^^^^^^^ ^^ the rug
# ^^^ ---     the ^^^^^^^^ ^^ the rug
#             ^^^ -------- ^^ the rug
#                          -- the ^^^
#                             ^^^ ---

def is_verb_phrase(pos_tags: list[str]):
    data = list(map(lambda x: 'VB' == x[:2], pos_tags))
    if any(data):
        index = data.index(True)
        return is_noun_phrase(data[index + 1:])
    return False


def is_prepositional_phrase(pos_tags: list[str]):
    data = list(map(lambda x: 'P' == x[:1], pos_tags))
    if any(data):
        index = data.index(True)
        return is_noun_phrase(data[index + 1:])
    return False


def is_noun_phrase(pos_tags: list[str]):
    data = list(map(lambda x: 'NP' == x[:2], pos_tags))
    if any(data):
        index = data.index(True)
        return is_prepositional_phrase(data[index + 1:]) or is_noun_phrase(data[index + 1:])
    return data == []


def is_adjective_phrase(pos_tags: list[str]):
    return any(map(lambda x: 'JJ' == x[:2], pos_tags))


def is_adverb_phrase(pos_tags: list[str]):
    return any(map(lambda x: 'RB' == x[:2], pos_tags))


def is_sentence(pos_tags: list[str]):
    return is_noun_phrase(pos_tags) and is_verb_phrase(pos_tags)


if __name__ == '__main__':
    test = 'Analysts said Mr. Stronach wants to resum a more influential role in running the company'
    all_pos_tags = nltk.pos_tag(test.split(' '))
    print(all_pos_tags)
    phrases = []
    for i in range(len(all_pos_tags) - 1, 0, -1):
        pass
    # print(is_noun_phrase())

# G = (T, C, N, S, L, R)
# G = (T, C, N, ROOT, L, R)
# G = (T, C, N, TOP, L, R)
# T set of Terminal Symbols
# C set of Pre-terminal Symbols
# N set of Non-terminal Symbols
# S is the start symbol S in N
# L is the lexicon; X -> x; X in P and x in T
# R is the grammar; X -> y; X in N and y in [N or C]


# G = (T, N, S, R, P)
# T set of Terminal Symbols
# N set of Non-terminal Symbols
# S is the start symbol S in N
# R is the set of rules; X -> y; X in N and y in [N or C]
# P is a probability function; P: R -> [0,1];
#           For all X in N sum[P(X-> y) for X->y in R] = 1
# Sum[P(y) for y in T*]= 1
