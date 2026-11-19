# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y).
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Yogalakshmi M
RegisterNumber:  212224020061
*/
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)

## Output:
<img width="1655" height="601" alt="image" src="https://github.com/user-attachments/assets/7d0f8a69-57d2-435a-a328-4821d5b2e649" />
<img width="1502" height="563" alt="image" src="https://github.com/user-attachments/assets/03570576-b484-49c4-82a7-44ce50742692" />
FIT TRANSFORM
<img width="1442" height="968" alt="image" src="https://github.com/user-attachments/assets/a70209a6-4c7d-4714-9a1f-04e3effb7c3e" />
LOGISTIC REGRESSION
<img width="1628" height="601" alt="image" src="https://github.com/user-attachments/assets/6faca918-698b-4489-973b-428865eceed4" />
ACCURACY SCORE
<img width="1745" height="376" alt="image" src="https://github.com/user-attachments/assets/c67f49ad-7bf7-459f-a1e1-709503b09867" />
CONFUSION MATRIX
<img width="1593" height="412" alt="image" src="https://github.com/user-attachments/assets/79dd3f00-80c7-445a-9451-dc75a8a24e31" />
CLASSIFICATION REPORT & PREDICTION
<img width="1620" height="818" alt="image" src="https://github.com/user-attachments/assets/3e659825-0aef-492a-b67f-ba565d83a23c" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
