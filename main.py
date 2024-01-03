import streamlit as st
# import pandas as pd
# import mitosheet as mt
# from mitosheet.streamlit.v1 import spreadsheet
# import matplotlib.pyplot as plt

###
# st.set_page_config(layout="wide")
st.title('Diabetes Health Indicators')
# CSV_URL = '/workspaces/psbnd2/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
# new_dfs, code = spreadsheet(CSV_URL)
# st.write(new_dfs)
# st.code(code)
###

# Judul
st.write("Hi! Please fill the form below.")

###
df = pd.read_csv('https://raw.githubusercontent.com/danisnurman/psbnd2/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
# st.dataframe(df, use_container_width=True)
df.dropna(inplace=True)
df.isnull().sum()
feature_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity']
df = df[feature_cols]
st.write(feature_cols)
X = df.drop(columns='Diabetes_binary')
y = df.Diabetes_binary

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
st.write("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

predict_result = clf.predict([[0,0,53,0,0]])
st.write(predict_result)

###
bpVal = st.number_input(label="High BP?", min_value=0, max_value=1)
cholVal = st.number_input(label="Cholesterol Total", min_value=10, max_value=500)
bmiVal = st.number_input(label="Body Mass Index", min_value=10, max_value=100)
smokerVal = st.number_input(label="Smoker?", min_value=0, max_value=1)
physActVal = st.number_input(label="Physical Activity?", min_value=0, max_value=1)
###

st.write(bpVal, cholVal, bmiVal, smokerVal, physActVal)
cholStatus = 0

if(cholVal>=10 and cholVal<=200):
    cholStatus = 0
elif(cholVal>200):
    cholStatus = 1

st.write("Cholesterol status: ", cholStatus)

predict_result = clf.predict([[bpVal, cholStatus, bmiVal, smokerVal, physActVal]])
if(predict_result==0):
    st.write("Diabetes status: Not Risk")
else:
    st.write("Diabetes status: Risk!")
# st.write(predict_result)
###