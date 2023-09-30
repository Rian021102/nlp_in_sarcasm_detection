from datapipn import datapipeline
from cleantext import TextPreprocessor
from tokenizesplit import tokenize_pad_split
from modeltrain import build_model_function, tune_hyperparameters_function, train_best_model_function, evaluate_model_function

filepath = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/Train_Data.csv'
df = datapipeline(filepath)
preprocessor = TextPreprocessor()
df['headline'] = df['headline'].apply(lambda x: preprocessor.preprocess_text(x))
X_train, X_test, y_train, y_test = tokenize_pad_split(df)

# Perform hyperparameter tuning and training
train_best_model_function(X_train, y_train, X_test, y_test)
