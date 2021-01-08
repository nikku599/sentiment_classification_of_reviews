
#importing the main libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data
df=pd.read_csv("Restaurant_Reviews.csv")
df.head()


#cleaning the data and redundancy

#removing the stopwords('a','an','at',etc)
from nltk.corpus import stopwords
Stop_words=stopwords.words('English')
Stop_words.remove('not')


#Stemming
#Same words might be present in different forms in reviews.
#Consider this example:
#I Loved the food.
#I love the food.
#Meaning of both the senteance is same. 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

#removing punctuation,stopwords and stemming
import re

rev_arr=[]
for i in range(len(df)):
    review=re.sub('[^a-zA-Z]',' ',df.iloc[i,0])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(Stop_words)]
    review=" ".join(review)
    rev_arr.append(review)

#reviews and final reviews after cleaning
print(list(df.Review)[:10])
print(rev_arr[:10])
    
#Creating Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)    
#defining independent and dependent variable
X=cv.fit_transform(rev_arr).toarray()
Y=df.iloc[:,1].values

#spiltting into training and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y, test_size=0.25,random_state=42)

#understanding the test case
print(len(xtest))
print(len(ytest))

count=0
for i in ytest:
    if i==1:
        count+=1
print(count)

#implementing the NaiveBayes model
from sklearn.naive_bayes import GaussianNB
model_nb=GaussianNB()
model_nb.fit(xtrain,ytrain)
ypred_nb=model_nb.predict(xtest)

#checking the accuracy score of naive bayes
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred_nb))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, ypred_nb))


#implementing the decision tree model
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(criterion="gini",random_state=42)    
model_dt.fit(xtrain,ytrain)    
ypred_dt=model_dt.predict(xtest)

#checking the accuracy score of decision tree
print(accuracy_score(ytest,ypred_dt))
print(confusion_matrix(ytest, ypred_dt))



#implementing knn
from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier()
model_knn.fit(xtrain,ytrain)
ypred_knn=model_knn.predict(xtest)

print(accuracy_score(ytest, ypred_knn))
print(confusion_matrix(ytest, ypred_knn))

#implementing the logisticRegression
from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression()
model_lr.fit(xtrain,ytrain)
ypred_lr=model_lr.predict(xtest)

#checking the accuracy score of Logistic Regression
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred_lr))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, ypred_lr))



