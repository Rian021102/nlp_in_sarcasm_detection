from datapipn import datapipeline
from cleantext import TextPreprocessor
from tokenizesplit import tokenize_pad_split
from modeltrain import trainmodel

filepath = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/train.csv'
df = datapipeline(filepath)
preprocessor = TextPreprocessor()
df['headline'] = df['headline'].apply(lambda x: preprocessor.preprocess_text(x))
X_train, X_test, y_train, y_test = tokenize_pad_split(df)

# Perform hyperparameter tuning and training
model=trainmodel(X_train, y_train, X_test, y_test)
