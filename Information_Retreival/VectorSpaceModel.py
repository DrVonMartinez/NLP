import glob
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd


class VectorSpaceModel:
    def __init__(self):
        self.__df_category = ['n', 't', 'p']
        self.__tf_category = ['n', 'l', 'a', 'b']
        self.__normalization_category = ['n', 'c']
        self.__document_term_frequency = 'n'
        self.__document_document_frequency = 'n'
        self.__document_normalization = 'n'
        self.__query_term_frequency = 'n'
        self.__query_document_frequency = 'n'
        self.__query_normalization = 'n'
        self.__document_matrix = pd.DataFrame()

    def __str__(self):
        ddd = self.__document_term_frequency + self.__document_document_frequency + self.__document_normalization
        qqq = self.__query_term_frequency + self.__query_document_frequency + self.__query_normalization
        return ddd + '.' + qqq

    @property
    def data(self):
        return self.__document_matrix

    @property
    def smart(self):
        return str(self)

    @smart.setter
    def smart(self, value):
        self.__document_term_frequency = value[0]
        self.__document_document_frequency = value[1]
        self.__document_normalization = value[2]
        self.__query_term_frequency = value[4]
        self.__query_document_frequency = value[5]
        self.__query_normalization = value[6]

    @property
    def document_tf(self):
        return self.__document_term_frequency

    @document_tf.setter
    def document_tf(self, tf: str):
        if tf in self.__tf_category:
            self.__document_term_frequency = tf
        else:
            raise NotImplementedError(tf + ' is not in the implemented set')

    @property
    def document_normalization(self):
        return self.__document_normalization

    @document_normalization.setter
    def document_normalization(self, n: str):
        if n in self.__normalization_category:
            self.__document_normalization = n
        else:
            raise NotImplementedError(n + ' is not in the implemented set')

    @property
    def document_df(self):
        return self.__document_term_frequency

    @document_df.setter
    def document_df(self, df: str):
        if df in self.__df_category:
            self.__document_document_frequency = df
        else:
            raise NotImplementedError(df + ' is not in the implemented set')

    @property
    def query_tf(self):
        return self.__query_term_frequency

    @query_tf.setter
    def query_tf(self, tf: str):
        if tf in self.__tf_category:
            self.__query_term_frequency = tf
        else:
            raise NotImplementedError(tf + ' is not in the implemented set')

    @property
    def query_normalization(self):
        return self.__query_normalization

    @query_normalization.setter
    def query_normalization(self, n: str):
        if n in self.__normalization_category:
            self.__query_normalization = n
        else:
            raise NotImplementedError(n + ' is not in the implemented set')

    @property
    def query_df(self):
        return self.__query_term_frequency

    @query_df.setter
    def query_df(self, df: str):
        if df in self.__df_category:
            self.__query_document_frequency = df
        else:
            raise NotImplementedError(df + ' is not in the implemented set')

    def __tf(self, term, document, tf_const):
        tf = self.__document_matrix[document][term]
        if tf_const == 'n':
            return tf
        elif tf_const == 'l':
            return 1 + np.log(tf)
        elif tf_const == 'a':
            return 0.5 + (0.5 * tf) / np.max(tf)
        elif tf_const == 'b':
            return 1 if tf > 0 else 0
        else:
            raise NotImplementedError('TF ' + tf_const)

    def __df(self, term, df_const):
        df = sum(self.__document_matrix.loc[term] > 0) - 1
        n = len(self)
        if df_const == 'n':
            return 1 if df > 0 else 0
        elif df_const == 't':
            return np.log(n / df)
        elif df_const == 'p':
            return np.max(0, np.log((n - df) / df))
        else:
            raise NotImplementedError('DF ' + df_const)

    def __normalization(self, document, n_const):
        tf = self.__document_matrix[document]
        if n_const == 'n':
            return 1
        elif n_const == 'c':
            return 1 / np.sqrt(np.sum([w ** 2 for w in tf]))
        else:
            raise NotImplementedError('Normalization ' + n_const)

    def __len__(self):
        return len(self.__document_matrix.columns)

    def query(self, query: str, k: int = 10):
        stemmer = PorterStemmer()
        words = word_tokenize(query)
        stemmed_queries = [stemmer.stem(word) for word in words]
        c = dict(Counter(stemmed_queries))
        df = pd.DataFrame.from_dict(c, orient='index', columns=['Query'])
        self.__document_matrix = pd.concat([self.__document_matrix, df], axis=1).fillna(0)
        query_score = self.__score(stemmed_queries, 'Query', 'Query')
        # print(query_score)
        self.__document_matrix['Query'] = np.nan
        columns = np.asarray(list(self.__document_matrix.columns)[:-1])
        scores = np.array([np.dot(self.__score(stemmed_queries, doc, 'Doc'), query_score) for doc in columns])
        ordered = np.argsort(scores)
        # sorted_scores = scores[ordered]
        ordered_columns = columns[ordered]
        # print(ordered_columns)
        # print(sorted_scores)
        # return sorted_scores[:k], ordered_columns[:k]
        return ordered_columns[:k]

    def __score(self, query: list[str], document: str, style: str):
        words = (self.__document_matrix[document][self.__document_matrix[document] > 0]).index
        intersect = set(query).intersection(set(words))
        if style == 'Doc':
            df = self.__document_document_frequency
            tf = self.__document_term_frequency
            nm = self.__document_normalization
        else:
            df = self.__query_document_frequency
            tf = self.__query_term_frequency
            nm = self.__query_normalization
        return np.sum([self.__tf(t, document, tf) * self.__df(t, df) * self.__normalization(document, nm)
                       for t in intersect])

    def add_file(self, file_path: str):
        file_name = [file_path[:-4]]
        stemmer = PorterStemmer()
        with open(file_path, 'r') as reader:
            words = word_tokenize(' '.join(reader.readlines()))
            stemmed_words = [stemmer.stem(word) for word in words]
            c = dict(Counter(stemmed_words))
            tf = pd.DataFrame.from_dict(c, orient='index', columns=file_name)
        self.__document_matrix = pd.concat([self.__document_matrix, tf], axis=1).fillna(0)
        self.__document_matrix.sort_values(list(self.__document_matrix.columns), ascending=False, inplace=True)

    def load(self, file_path):
        self.__document_matrix = pd.read_csv(file_path, index_col=0)

    def save(self, file_path):
        self.__document_matrix.to_csv(file_path, columns=sorted(self.__document_matrix.columns))


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
    # print(index_set)
    for i in range(len(index_set) - 1):
        with open('complete_shakespeare.txt', 'r') as reader:
            lines = reader.readlines()[index_set[i]: index_set[i + 1]]
            title = lines[0][:-1].replace("'", '')
            with open(title.title() + '.txt', 'w+') as writer:
                for line in lines:
                    writer.write(''.join([q if q.isalnum() or q == ' ' else '' for q in line]) + '\n')
            print(title, 'complete')


def train(vsm: VectorSpaceModel):
    files = list(glob.glob('*.txt'))
    files.remove('complete_shakespeare.txt')
    for file in files:
        vsm.add_file(file)
    print(len(vsm), len(files))
    vsm.save('Dataframe.csv')


if __name__ == '__main__':
    # extract_titles()
    ir = VectorSpaceModel()
    ir.smart = 'ltc.ltc'
    # train(ir)
    ir.load('Dataframe.csv')
    print(results := ir.query('to be or not to be'))
