import setup_texts as setup
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import abc


class VectorSpaceModel(abc.ABC):
    def __init__(self):
        self._df_category = ['n', 't', 'p']
        self._tf_category = ['n', 'l', 'a', 'b']
        self._normalization_category = ['n', 'c']
        self._document_term_frequency = 'n'
        self._document_document_frequency = 'n'
        self._document_normalization = 'n'
        self._query_term_frequency = 'n'
        self._query_document_frequency = 'n'
        self._query_normalization = 'n'
        self._document_matrix = pd.DataFrame()

    def __str__(self):
        ddd = self._document_term_frequency + self._document_document_frequency + self._document_normalization
        qqq = self._query_term_frequency + self._query_document_frequency + self._query_normalization
        return '{ddd}.{qqq}'.format(ddd=ddd, qqq=qqq)

    @property
    def data(self):
        return self._document_matrix

    @property
    def smart(self):
        return str(self)

    @smart.setter
    def smart(self, value: str):
        if len(value) == 7 and value[3] == '.':
            self.document_tf = value[0]
            self.document_df = value[1]
            self.document_normalization = value[2]
            self.query_tf = value[4]
            self.query_df = value[5]
            self.query_normalization = value[6]

    @property
    def document_tf(self):
        return self._document_term_frequency

    @document_tf.setter
    def document_tf(self, tf: str):
        if tf in self._tf_category:
            self._document_term_frequency = tf
        else:
            raise NotImplementedError(tf + ' is not in the implemented set')

    @property
    def document_normalization(self):
        return self._document_normalization

    @document_normalization.setter
    def document_normalization(self, n: str):
        if n in self._normalization_category:
            self._document_normalization = n
        else:
            raise NotImplementedError(n + ' is not in the implemented set')

    @property
    def document_df(self):
        return self._document_term_frequency

    @document_df.setter
    def document_df(self, df: str):
        if df in self._df_category:
            self._document_document_frequency = df
        else:
            raise NotImplementedError(df + ' is not in the implemented set')

    @property
    def query_tf(self):
        return self._query_term_frequency

    @query_tf.setter
    def query_tf(self, tf: str):
        if tf in self._tf_category:
            self._query_term_frequency = tf
        else:
            raise NotImplementedError(tf + ' is not in the implemented set')

    @property
    def query_normalization(self):
        return self._query_normalization

    @query_normalization.setter
    def query_normalization(self, n: str):
        if n in self._normalization_category:
            self._query_normalization = n
        else:
            raise NotImplementedError(n + ' is not in the implemented set')

    @property
    def query_df(self):
        return self._query_term_frequency

    @query_df.setter
    def query_df(self, df: str):
        if df in self._df_category:
            self._query_document_frequency = df
        else:
            raise NotImplementedError(df + ' is not in the implemented set')

    def _tf(self, term, document, tf_const):
        tf = self._document_matrix[document][term]
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

    def _df(self, term, df_const):
        df = sum(self._document_matrix.loc[term] > 0) - 1
        n = len(self)
        if df_const == 'n':
            return 1 if df > 0 else 0
        elif df_const == 't':
            return np.log(n / df)
        elif df_const == 'p':
            return np.max(0, np.log((n - df) / df))
        else:
            raise NotImplementedError('DF ' + df_const)

    def _normalization(self, document, n_const):
        tf = self._document_matrix[document]
        if n_const == 'n':
            return 1
        elif n_const == 'c':
            return 1 / np.sqrt(np.sum([w ** 2 for w in tf]))
        else:
            raise NotImplementedError('Normalization ' + n_const)

    def __len__(self):
        return len(self._document_matrix.columns)

    @staticmethod
    def _normalize(stream: str):
        stemmer = PorterStemmer()
        words = word_tokenize(stream)
        return [stemmer.stem(word) for word in words]

    def _to_dataframe(self, stream: list[str], column_name: list[str]):
        c = dict(Counter(stream))
        df = pd.DataFrame.from_dict(c, orient='index', columns=column_name)
        self._document_matrix = pd.concat([self._document_matrix, df], axis=1).fillna(0)

    def query(self, query: str, k: int = 10):
        query_stream = self._normalize(query)
        self._to_dataframe(query_stream, ['Query'])
        query_score = self._score(query_stream, 'Query', 'Query')
        self._document_matrix.drop(columns=['Query'], inplace=True)
        columns = np.asarray(list(self._document_matrix.columns)[:-1])
        scores = np.array([np.dot(self._score(query_stream, doc, 'Doc'), query_score) for doc in columns])
        ordered = np.argsort(scores)
        ordered_columns = columns[ordered]
        return ordered_columns[:k]

    def _score(self, query: list[str], document: str, style: str):
        words = (self._document_matrix[document][self._document_matrix[document] > 0]).index
        intersect = set(query).intersection(set(words))
        if style == 'Doc':
            df = self._document_document_frequency
            tf = self._document_term_frequency
            nm = self._document_normalization
        else:
            df = self._query_document_frequency
            tf = self._query_term_frequency
            nm = self._query_normalization
        return np.sum([self._tf(t, document, tf) * self._df(t, df) * self._normalization(document, nm)
                       for t in intersect])

    def load(self, file_path):
        self._document_matrix = pd.read_csv(file_path, index_col=0)

    def save(self, file_path):
        self._document_matrix.to_csv(file_path, columns=sorted(self._document_matrix.columns))


class DocumentVSM(VectorSpaceModel):
    def __init__(self):
        super().__init__()

    def add_file(self, file_path: str):
        file_name = [file_path[:-4]]
        with open(file_path, 'r') as reader:
            words = ' '.join(reader.readlines())
        word_stream = self._normalize(words)
        self._to_dataframe(word_stream, file_name)

    def add_document(self, words: str or list[str], document_name: str):
        if isinstance(words, list):
            words = ' '.join(words)
        word_stream = self._normalize(words)
        self._to_dataframe(word_stream, [document_name])


class WordVSM(VectorSpaceModel):
    def __init__(self):
        super().__init__()

def train(vsm: DocumentVSM):
    files = setup.group()
    for file in files:
        vsm.add_file(file)
    print(len(vsm), len(files))
    vsm.save('Dataframe.csv')


if __name__ == '__main__':
    # extract_titles()
    ir = DocumentVSM()
    ir.smart = 'ltc.ltc'
    # train(ir)
    ir.load('Dataframe.csv')
    print(results := ir.query('to be or not to be'))
