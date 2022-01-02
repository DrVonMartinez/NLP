from nltk.corpus import brown


def is_word(word):
    char_list = [".", "-", '`', ',', '?', ':', "'"]
    while char_list:
        word = word.replace(char_list.pop(0), '')
    return str(word).isalnum()


words = list(map(str.lower, brown.words()))
with open('training_data.txt', 'w+') as writer:
    stopwords = list(filter(lambda x: not is_word(x), words))
    old_index = 0
    i = 0
    while stopwords:
        if i == 1:
            assert False
        i += 1
        next_stop = stopwords.pop(0)
        index = str(words[old_index:]).find(next_stop)
        for j in range(old_index, index - 3):
            if any(map(lambda x: x in [".", "-", '`', ',', '?', ':', '"'], ' '.join(words[j: j + 2]))):
                continue
            writer.write(' '.join(words[j: j + 2]) + '\n')
            print(' '.join(words[j: j + 2]))
        old_index = index
