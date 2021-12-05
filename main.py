import pandas as pd

df = pd.read_csv('Negative Sentences.csv')
print(df)
text = ' '.join(map(str, df['THE NEGATIVE SENTENCES ARE:']))
print(text)
