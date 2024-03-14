from datapipn import datapipeline, TextPreprocessor
from tokenizesplit import tokenize_pad_split, calculate_class_weights
from trainlstm import trainmodel
from bidrlstm import trainmodelbid

def main():
    filepath = '/Users/rianrachmanto/miniforge3/project/sarcastic_detection/data/train.csv'
    df = datapipeline(filepath)
    preprocessor = TextPreprocessor()
    df['headline'] = df['headline'].apply(lambda x: preprocessor.preprocess_text(x))
    X_train, X_test,y_train,y_test = tokenize_pad_split(df)
    model=trainmodelbid(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
