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

5.Display the results. 

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nivetha A
RegisterNumber:212222230101  
```
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()
```
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
```
data1.isnull()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
```
x=data1.iloc[:,:-1]
x
```
```
y=data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### Placement data
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/12905eb0-430c-469c-b47a-32dcda4acfec)


### Salary data
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/95ee625d-3079-42c1-a6ad-fe760996e5e8)

### checking the null function
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/ccb69f58-e417-4daa-86e2-7c1465b2e2a3)


### Data duplicate
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/04a3dfa4-7f04-4f7d-9271-01339a648e19)


### Print data
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/918db73b-cbf6-4703-a335-893d69944fe9)


### Data status
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/15204183-48cd-4ca1-b204-1696368b4f0b)


### y prediction array
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/820a6874-4a29-4dc5-aba9-bbc433aa7e44)


### Accuracy value
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/e166f610-2ecb-4bf7-9c5f-f68dcdd7a3fd)


### Classification report
![image](https://github.com/nivetharajaa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120543388/0bb9f9f7-d592-4837-846b-b943396d8bb9)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
