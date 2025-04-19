# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KUKKADAPU CHARAN TEJ
RegisterNumber: 212224040167
*/
```
```py
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
TOP 3 ELEMENTS

![0 1](https://github.com/user-attachments/assets/a39ef24a-501b-412c-943e-04ef64e3d232)

![0 2](https://github.com/user-attachments/assets/d03d2862-22f7-4989-802d-258de923e459)

DATA DUPLICATE

![0 3](https://github.com/user-attachments/assets/9588f01c-aff4-4fe5-8c20-27f2ef300004)

PRINT DATA

![0 4](https://github.com/user-attachments/assets/210ff5ae-98a1-40d7-af7a-724a58aef377)

DATA STATUS

![0 5](https://github.com/user-attachments/assets/162a3e99-e749-4d2a-bfaf-2bdbe5bcf554)


Y-PREDICTION ARRAY

![0 6](https://github.com/user-attachments/assets/a7bb117e-d8a2-4bf9-8cbc-a86c5311b984)

CONFUSION ARRAY

![0 7](https://github.com/user-attachments/assets/fecb3d1e-bbcb-4105-99f6-3587fc1f09f0)


ACCURACY VALUE

![0 8](https://github.com/user-attachments/assets/454a8eb0-1593-4cd0-be9e-b7623bb1e5bb)


CLASSIFICATION REPORT

![0 9](https://github.com/user-attachments/assets/e080dff6-89a1-419f-b6d0-63cf8f44146d)


PREDICTION OF LR

![0 10](https://github.com/user-attachments/assets/c13fb6fa-b906-4c2a-ab16-894bb1c0e0a2)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
