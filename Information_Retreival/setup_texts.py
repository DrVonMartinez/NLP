import glob


def extract_titles():
    titles = ["ALLS WELL THAT ENDS WELL", "ANTHONY AND CLEOPATRA", "AS YOU LIKE IT",
              "THE COMEDY OF ERRORS", "CORIOLANUS", "CYMBELINE",
              "HAMLET, PRINCE OF DENMARK", "THE FIRST PART OF KING HENRY IV",
              "THE SECOND PART OF KING HENRY IV", "THE LIFE OF KING HENRY V",
              "THE FIRST PART OF KING HENRY VI", "THE SECOND PART OF KING HENRY VI",
              "THE THIRD PART OF KING HENRY VI", "THE FAMOUS HISTORY OF THE LIFE OF KING HENRY VIII",
              "THE LIFE AND DEATH OF KING JOHN", "JULIUS CAESAR", "KING LEAR", "LOVE'S LABOUR'S LOST",
              "MACBETH", "MEASURE FOR MEASURE", "THE MERCHANT OF VENICE", "THE MERRY WIVES OF WINDSOR",
              "A MIDSUMMER-NIGHT'S DREAM", "MUCH ADO ABOUT NOTHING", "OTHELLO, THE MOOR OF VENICE",
              "PERICLES, PRINCE OF TYRE", "THE TRAGEDY OF KING RICHARD II", "THE TRAGEDY OF KING RICHARD III",
              "ROMEO AND JULIET", "THE TAMING OF THE SHREW", "THE TEMPEST",
              "TIMON OF ATHENS", "TITUS ANDRONICUS", "TROILUS AND CRESSIDA",
              "TWELFTH-NIGHT; OR WHAT YOU WILL", "THE TWO GENTLEMEN OF VERONA", "THE WINTERS TALE"]
    index_set = []
    for title in titles:
        temp_title = title + '\n'
        with open('complete_shakespeare.txt', 'r') as reader:
            lines = reader.readlines()
            index_set.append(lines.index(temp_title))
    index_set.sort()
    for i in range(len(index_set) - 1):
        with open('complete_shakespeare.txt', 'r') as reader:
            lines = reader.readlines()[index_set[i]: index_set[i + 1]]
            title = lines[0][:-1].replace("'", '')
            with open('Texts\\' + title.title() + '.txt', 'w+') as writer:
                for line in lines:
                    writer.write(''.join([q if q.isalnum() or q == ' ' else '' for q in line]) + '\n')
            print(title, 'complete')


def group():
    return list(glob.glob('Texts\\*.txt'))