import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/content/Churn_Modelling.csv')

data.head()
data.shape
data.isnull().sum()
data.describe()
data.columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)
data['Geography'].unique()
data = pd.get_dummies(data,drop_first=True)
data.head()
data['Exited'].value_counts()

sns.countplot(data['Exited'])
plt.show()

X = data.drop(['Exited'],axis = 1)
y=data['Exited']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
sc =StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)

joblib.dump(rf,'churn_predict_model')
model = joblib.load('churn_predict_model')
data.columns
model.predict([[619,42,2,0.0,0,0,0,101348.88,0,0,0]])
