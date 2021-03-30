import sys

import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import pickle


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
stop_words = set(stopwords.words('english'))


def load_data(database_filepath):
    """
    Load categorized messages from database.
    
    ARGUMENT:
    database_filepath - SQLite database
    
    RETURN VALUES:
    
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('categorized_messages', con=engine)
    
    X = df['message']
    y = df.iloc[:, 4:].values
    category_names = df.columns[4:]
    
    return X, y, category_names


def tokenize(text):
    """
    Normalize, tokenize, and lemmatize text.
    
    ARGUMENT:
    text - string to be tokenized
    
    RETURN VALUE:
    cleaned_tokens - list of cleaned tokens
    """
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    words = word_tokenize(text)
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
    
    return cleaned_tokens


def build_model():
    """ 
    Create a ML Pipeline and optimize its parameters with GridSearch.
    
    ARGUMENT:
    None
    
    RETURN VALUE:
    model - initiated and optimized model 
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    # Uncommenting more parameters will give better exploring power but 
    # will increase processing time in a combinatorial way
    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__bootstrap': (True, False),
        #'clf__estimator__criterion': ('gini', 'entropy')
    }

    # Change cv=2 to cv=None for a 5 fold cross validation
    cv = GridSearchCV(pipeline, param_grid=parameters, 
                      scoring='f1_macro', verbose=3, cv=2, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print out accuracy, precision, recall, and f1 score for the model.
    
    ARGUMENTS:
    model - model fitted on training dataset
    X_test - test feature dataset
    Y_test - test target dataset
    category_names - target variables
    
    RETURN VALUE:
    None
    """
    
    Y_pred = model.predict(X_test)
    Y_pred_ = pd.DataFrame(Y_pred)
    Y_test_ = pd.DataFrame(Y_test)
    
    for i, col in enumerate(category_names):
        if Y_test_[i].sum() == 0:
            print(col + ' no positive labels in test dataset\n')
        else:
            accuracy = int(round(100 * accuracy_score(Y_test_[i], Y_pred_[i])))
            print(f'{col} | accuracy: {accuracy}%')
            print(classification_report(Y_test_[i], Y_pred_[i]) + '\n')


def save_model(model, model_filepath):
    """
    Save model to pickle file.
    
    ARGUMENTS:
    model - trained ML model
    model_filepath - file path
    
    RETURN VALUE:
    None
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()