import numpy as np
import pandas as pd
import re
import nltk
import stopwords

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#print(stopwords.get_stopwords('english'))
news_dataset = pd.read_csv('train.csv')
print(news_dataset.shape)
print(news_dataset.head())

print(news_dataset.isnull().sum())

news_dataset=news_dataset.fillna('')

news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

print(news_dataset['content'])

x= news_dataset.drop(columns='label',axis=1)
y= news_dataset['label']

print(x)
print(y)

port_strem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_strem.stem(word) for word in stemmed_content if not word in stopwords.get_stopwords('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming)
print(news_dataset['content'])

x= news_dataset['content'].values
y= news_dataset['label'].values

print(x)

print(y)

print(y.shape)


vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x =vectorizer.transform(x)

print(x)

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

model = LogisticRegression()

model.fit(x_train,y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


print('Accuracy score of the training data :' ,training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)

print("Accuracy Score Of the Test Data : ", test_data_accuracy)


x_new = x_test[0]

prediction = model.predict(x_new)
print(prediction)


if (prediction[0] == 0):
    print('This News  is Real')
else:
    print('This News is Fake')

print(y_test[6])
