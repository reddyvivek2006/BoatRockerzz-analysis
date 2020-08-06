from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

data=pd.read_csv("C:/Users/VIVEK REDDY S/Desktop/AllProductReviews.csv")
#data=data[data.ReviewStar!=3]
corpus=[]
print(data.head(3))
#data=data[data.ReviewBody.str.contains(pat='[a-z A-Z]*',regex=True)==True]
data.dropna()
#print(data.head(7))
print(data.shape)
k=0
i=1
pattern='[^ a-zA-Z0-9]*'
print(data.ReviewBody[1])
while(k<data.shape[0]-1):
     a=re.sub(pattern,'',data['ReviewBody'][i])
     k=k+1
     i=i+1
print(data.shape)
y=[]
for i in range(0,data.shape[0]):
     if(data.ReviewStar[i]!=3):
         review = re.sub('[^a-z A-Z]', ' ', data['ReviewBody'][i])
         
         review=data['ReviewBody'][i].lower()
         
         review=review.split()
         
         ps=PorterStemmer()
         
         review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

         review=' '.join(review)
         corpus.append(review)
         print(review)
     if(data.ReviewStar[i]>=4):
          y.append(1)
     elif(data.ReviewStar[i]<=2):
          y.append(0)
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(y_pred)
print(corpus)
print(len(y))
