import sys
import pandas as pd
import re
import nltk
import pickle
import numpy as np
nltk.download(['wordnet', 'punkt', 'stopwords'])
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
   
    # load data from database
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql_table('DisasterResponse',engine)
    
    #The messages col
    X = df ['message'] 
    #Target variables 
    y = df.iloc[:,4:] 
    
    return X, y

def tokenize(text):
        
    #Removing punctuation characters and converting the text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Spliting text into words using NLTK
    words = word_tokenize(text)
    
    #Removing stop words
    words = [w for w in words 
             if w not in stopwords.words("english")]
    
    #Reducing words to their root form
    stemmed = [PorterStemmer().stem(w) for w in words]

    return stemmed


def build_model():
    #Building the machine learning pipeline
    #I chose to go with Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #Identifing the parameters 
    parameters = {
            'clf__estimator__warm_start': (True, False),
            'tfidf__smooth_idf': (True, False)
    }

    return GridSearchCV(pipeline, param_grid=parameters)
    
    


def evaluate_model(model, X_test, Y_test):
    
    
    y_pred = model.predict(X_test)
    
    #Counter start from 0
    counter=0

    #For loop start form the first col in y_tset 
    for col in Y_test:

        #Printing col name
        print(col) 

        #Printing classification report for each class
        print(classification_report(Y_test[col], y_pred[:,counter]))

        #Increasing the counter
        counter=counter+1

    #Calculating the accuracy for each calss    
    accuracy_class = (y_pred == Y_test).mean()
    print(accuracy_class)

    #Calculating the accuracy for the model    
    accuracy_model = (y_pred == Y_test.values).mean()
    print("The model accuracy is:")
    print(accuracy_model)
    


def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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