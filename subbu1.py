import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
sep =":::"

test_data=pd.read_csv(r"C:\Users\Jay\Downloads\archive\Genre Classification Dataset\test_data_solution.txt",sep=':::',engine='python')
test_data_solution=pd.read_csv(r"C:\Users\Jay\Downloads\archive\Genre Classification Dataset\test_data_solution.txt",sep=':::',engine='python')
train_data=pd.read_csv(r"C:\Users\Jay\Downloads\archive\Genre Classification Dataset\train_data.txt",sep=":::",engine='python')


train_data.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

test_data.columns = ['ID', 'TITLE', 'GENRE','DESCRIPTION']



'''
print("Train Data Columns and Sample:")
print(train_data.columns)
print(train_data.head())

print("\nTest Data columns and sample:")
print(test_data.columns)
print(test_data.head())

print("\nTest Data Solution columns ans samnple")
print(test_data_solution.columns)
print(test_data_solution.head())
'''



plot_column='DESCRIPTION'
genre_column='GENRE'

train_data[plot_column] = train_data[plot_column].fillna('')
test_data[plot_column] = test_data[plot_column].fillna('')


label_encoder=LabelEncoder()
train_data['genre_encoded']=label_encoder.fit_transform(train_data[genre_column])


tfidf_vectorizer=TfidfVectorizer(max_features=5000,stop_words='english')
X_train_tfidf=tfidf_vectorizer.fit_transform(train_data[plot_column])
X_test_tfidf=tfidf_vectorizer.transform(test_data[plot_column])


X=X_train_tfidf
y=train_data['genre_encoded']

X_train,X_val,y_train,y_val=train_test_split(X,y, test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(X_train,y_train)

y_val_pred=log_reg.predict(X_val)

print('logistic regression')
print('accuracy:',accuracy_score(y_val,y_val_pred)*100)
print(classification_report(y_val,y_val_pred,target_names=label_encoder.classes_,zero_division=0))

from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()
nb.fit(X_train,y_train)

y_val_pred=nb.predict(X_val)

print('Naive Bayes')
print('accuracy_score',accuracy_score(y_val,y_val_pred)*100)
print(classification_report(y_val,y_val_pred,target_names=label_encoder.classes_,zero_division=0))

from sklearn.svm import LinearSVC

svc= LinearSVC(max_iter=1000)
svc.fit(X_train,y_train)

y_val_pred=svc.predict(X_val)

print('SUPPORT VECTOR MACHINE')
print('ACCURACY',accuracy_score(y_val,y_val_pred)*100)
print(classification_report(y_val,y_val_pred,target_names=label_encoder.classes_,zero_division=0))