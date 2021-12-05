# How many words?
# Lemma
# Wordform

# Type
# Token
# N = number of tokens
# V = vocabulary = set of types = |V|

# Google N-grams
#   1 trillion Tokens
#   13 million Types
# |V| > O(N^0.5)

# Maximum Matching

import re


def maximum_matching(string: str, wordlist: set[str]) -> list[str]:
    string = string.lower()
    words: list[str] = []
    pointer: int = 0
    while pointer < len(string):
        longest_word = ''
        for word in wordlist:
            if re.search(r'^' + word, string[pointer:]) and len(word) > len(longest_word):
                longest_word = word
        print(longest_word)
        words.append(longest_word)
        pointer += len(longest_word)
    return words


# Lemmatization
#   correct dictionary headword form
# Morphemes:
#   Stems
#   Affixes: adhered to stems

def porter_stemmer(words: list[str]):
    def stemming(word: str):
        # Plural
        if re.search(r'sses$', word):
            word = word[:-2]
        elif re.search(r'ies$', word):
            word = word[:-3] + 'i'
        elif re.search(r'ss$', word):
            pass
        elif re.search(r's$', word):
            word = word[:-1]
        # Action
        if re.search(r'.*[aeiou].*ing$', word.lower()):
            word = word[:-3]
        elif re.search(r'.*[aeiou].*ed$', word.lower()):
            word = word[:-2]
        return word

    print(list(map(stemming, words)))


porter_stemmer('caresses ponies caress cats walking sing plastered'.split(' '))
