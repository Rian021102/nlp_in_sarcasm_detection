from datapipn import datapipeline, TextPreprocessor
from tokenizesplit import tokenize_pad_split
from modeltrain import trainmodel

filepath = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/train.csv'
X_train, X_test, y_train, y_test = datapipeline(filepath)
preprocessor = TextPreprocessor()
X_train['headline'] = X_train['headline'].apply(lambda x: preprocessor.preprocess_text(x))
X_test['headline'] = X_test['headline'].apply(lambda x: preprocessor.preprocess_text(x))
X_train, X_test = tokenize_pad_split(X_train, X_test)

# Perform hyperparameter tuning and training
model=trainmodel(X_train, y_train, X_test, y_test)
