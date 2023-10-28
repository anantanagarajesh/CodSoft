import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mail_dataset = pd.read_csv('/content/mail_data.csv')

mail_dataset.head()
mail_dataset.shape

Message =['Message']
for Message in Message:
  mail_dataset[Message]=mail_dataset[Message].fillna('')

mail_dataset.loc[mail_dataset['Category']=='spam','Category',]=0
mail_dataset.loc[mail_dataset['Category']=='ham','Category',]=1

X=mail_dataset['Message']
Y=mail_dataset['Category']

#splitting the dataset into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
# converting to feature vectors
feature_extraction =TfidfVectorizer(min_df=1,stop_words='english',lowercase=1)

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

#model training
model=LogisticRegression()
model =model.fit(X_train_features,Y_train)

#evaluation
pred=model.predict(X_train_features)
accuracy=accuracy_score(pred,Y_train)
print(accuracy)

#testing
input=["Free entry in 2 a wkly A to 87121 to receive"]
prediction=model.predict(input_data)
print(prediction)
