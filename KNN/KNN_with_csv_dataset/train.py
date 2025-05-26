import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from KNN import KNN

df = pd.read_csv("exercise.csv")

df_encoded = df.copy()
le_diet = LabelEncoder()
le_time = LabelEncoder()
le_kind = LabelEncoder()

df_encoded['diet'] = le_diet.fit_transform(df_encoded['diet'])
df_encoded['time'] = le_time.fit_transform(df_encoded['time'])
df_encoded['kind'] = le_kind.fit_transform(df_encoded['kind'])

# print(le_diet.classes_)  # ['low fat', 'no fat']
# print(le_time.classes_)  # ['1 min', '15 min', '30 min']
# print(le_kind.classes_)

X = df_encoded[['pulse','time','kind']].values  # thêm .values để ép kiểu thành numpy array
y = df_encoded['diet'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)
clf = KNN(5)
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)
print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)