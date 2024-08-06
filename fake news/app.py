# Importing Necessary Libraries
from re import X
import numpy as np 
import pandas as pd 
from pandas_profiling import ProfileReport
import csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import nltk
from joblib import load  # This should be at the top of your file
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already done)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
#Create a svm Classifier
model = svm.SVC(kernel='linear') # Linear Kernel
clf_poly=clf = svm.SVC(kernel='poly') # Polynomial kernel
nltk.download("stopwords")
nltk.download("wordnet")
import os

webapp=Flask(__name__)


@webapp.route('/')
def index():
    return render_template('index.html')

@webapp.route('/about')
def about():
    return render_template('about.html')

@webapp.route('/load',methods=["GET","POST"])
def load():
    if request.method=="POST":
        file1=request.files['file1']
        file2 = request.files['file2']

        # file1.save(os.path.join(app.root_path, 'static/customlogos/logo.png'))

        global df1,dataset1,df2,dataset2
        df1=pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        dataset1=df1.head(100)
        dataset2=df2.head(100)
        msg='Real and Fake News  Data Loaded Successfully'
        return render_template('load.html',msg=msg)
    return render_template('load.html')

@webapp.route('/fakedata')
def fakedata():
    return render_template('fakedata.html', columns=dataset1.columns.values, rows=dataset1.values.tolist())

@webapp.route('/realdata')
def realdata():
    return render_template('realdata.html', columns=dataset2.columns.values, rows=dataset2.values.tolist())

def preprocess_data(df):
    # Convert text to lowercase
    df['text'] = df['text'].str.strip().str.lower()
    
    # Remove punctuation
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    
    # Remove numbers
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d+', '', x))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))
    
    return df

@webapp.route('/preprocess',methods=['POST','GET'])
def preprocess():
    global x, y, X_train, X_test, y_train, y_test, x_test, X_transformed, X_test_transformed, vec, df1, df2
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100

        df1 = df1[:2000]
        df2 = df2[:2000]
        df1['output'] = 1
        df2['output'] = 0

        df = [df1, df2]
        df = pd.concat(df)
        df = df[['text', 'output']]

        df = preprocess_data(df)

        x = df['text']
        y = df['output']
        x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

        # Use TfidfVectorizer
        vec = TfidfVectorizer(stop_words='english')
        x = vec.fit_transform(x).toarray()
        x_test = vec.transform(x_test).toarray()

        return render_template('preprocess.html', msg='Data Preprocessed and Trained Successfully')
    return render_template('preprocess.html')



@webapp.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
            rf = rf.fit(x,y)
            # Predicting the Test set results
            acc_rf = rf.score(x_test, y_test)*100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
            dt = dt.fit(x,y)
            acc_dt = dt.score(x_test, y_test)*100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s==3:
            # fit model no training data
            xgb = svm.SVC(kernel='linear')
            xgb = xgb.fit(x, y)
            # make predictions for test data
            acc_xgb = xgb.score(x_test, y_test)*100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(acc_xgb) + str('%')
            return render_template('model.html', msg=msg)
    return render_template('model.html')
'''
import joblib
@webapp.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        f1 = request.form['text']

        # Load the previously saved SVC model
        svc_model = joblib.load(r'News _dataset\svm_model.joblib')
        
        # The input needs to be in the same format as the model was trained on
        # Assuming necessary preprocessing was included in the model's pipeline
        # transformed_input = [f1]  # Example assumes model can handle raw input
        
        # Use the loaded model for prediction
        result = svc_model.predict(f1)
        
        if result[0] == 0:
            msg = 'The Entered Text Is A Fake News'
        else:
            msg = 'The Entered Text Is A Real News'
            
        return render_template('prediction.html', msg=msg)

    return render_template('prediction.html')
'''
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    
    return text

@webapp.route('/prediction',methods=['POST','GET'])
def prediction():
    global X_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)

        xgb = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=3, n_estimators=100, oob_score=True)
        xgb = xgb.fit(x, y)
        preprocessed_text = preprocess_text(f1)
    
        # Transform preprocessed text using the vectorizer
        transformed_input = vec.transform([preprocessed_text]).toarray()
        result = xgb.predict(transformed_input)
        print(result)
        if result==0:
            msg = 'The Entered Text Is A Fake News'
        else:
            msg= 'The Entered Text Is A Real News'
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')

@webapp.route('/news')
def news():
    return render_template('news.html')



if __name__=='__main__':
    webapp.run(debug=True)