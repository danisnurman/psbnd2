import streamlit as st
import pandas as pd
import mitosheet as mt
from mitosheet.streamlit.v1 import spreadsheet
import matplotlib.pyplot as plt

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
feature_cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack']
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

predict_result = clf.predict([[0,0,1,53,0,0,0]])
st.write(predict_result)

### Visualize
# from sklearn.tree import export_graphviz
# from six import StringIO
# from IPython.display import Image
# import pydotplus

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=
#                 ['bp', 'chol', 'cholcheck', 'bmi', 'smoker', 'stroke', 'heart'],
#                 class_names=['0', '1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())
###

###
# df2 = pd.DataFrame(
#     [
#         {"command": "st.selectbox", "rating": 4, "is_widget": True},
#         {"command": "st.balloons", "rating": 5, "is_widget": False},
#         {"command": "st.time_input", "rating": 3, "is_widget": True},
#     ]
# )
# st.dataframe(df2, use_container_width=True)
###

###
bpVal = st.number_input(label="High BP?", min_value=0, max_value=1)
cholVal = st.number_input(label="High Cholesterol?", min_value=0, max_value=1)
cholCheckVal = st.number_input(label="Cholesterol Check?", min_value=0, max_value=1)
bmiVal = st.number_input(label="Body Mass Index", min_value=10, max_value=100)
smokerVal = st.number_input(label="Smoker?", min_value=0, max_value=1)
strokeVal = st.number_input(label="Stroke?", min_value=0, max_value=1)
heartVal = st.number_input(label="Heart Disease or Attack?", min_value=0, max_value=1)
###

st.write(bpVal, cholVal, cholCheckVal, bmiVal, smokerVal, strokeVal, heartVal)

predict_result = clf.predict([[bpVal, cholVal, cholCheckVal, bmiVal, smokerVal, strokeVal, heartVal]])
if(predict_result==0):
    st.write(predict_result, "Not Risk")
else:
    st.write(predict_result, "Risk!")
# st.write(predict_result)
###