
# coding: utf-8

# ![Final Lesson Exercise](images/Banner_FEX.png)

# # Lesson #7: Supervised Learning - classification and regression
# ## Good Movies - The IMDb movie dataset

# ## About this assignment
# In this assignment, you will continue to explore information regarding good movies.<br/>
# 
# This time you will practice the classification and regression flow.<br />
# You will need to do the following:
# * Distinguish between classification and regression problems
# * Train classification and regression models and predict the test examples
# * Run evaluation for the classification and regression models
# * Use the tools you've got to manipulate features, in order to improve the results

# ## Preceding Step - import modules (packages)
# This step is necessary in order to use external packages. 
# 
# **Use the following libraries for the assignment, when needed**:

# In[ ]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP 

import os                       # for testing use only
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sklearn
from sklearn import linear_model, metrics, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, f1_score

#get_ipython().magic('matplotlib inline')


# ## 1. Load the dataset
# In this section you will need to load the dataset and split the dataset to feature vectors (X) and target (y).<br />
# 
# Note: the target could vary and represent a classification target or a regression target.

# ### 1. Instructions
# <u>method name</u>: <b>load_dataset</b>
# <pre>The following is expected:
# --- Complete the 'load_dataset' function to load the dataset 
#     from the csv, located in the 'file_name' parameter into a pandas dataframe.
# You should return a separate dataframe for the feature vectors 
#     and a series containing the corresponding target values, determined by the 'target_column' column.
# Notes: 
#       The 'X' dataframe should not include the 'target_column' column.
# </pre>
# <hr>
# Assuming 'X' represents the feature vector dataframe and 'y' represents the target values series,<br />
# The return statement should look similar to the following statement:<br />
# <b>return X, y</b>

# In[ ]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def load_dataset(file_name, target_column):
    X=pd.read_csv(file_name)
    y=X[target_column]
    X.drop(target_column, axis=1, inplace=True)
    return X, y


# In[ ]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
target_col_name = 'imdb_score'
X, y = load_dataset(file_name, target_col_name)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 1 (name: test1-1_load_dataset, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
target_col_name = 'imdb_score'

try:
    X, y = load_dataset(file_name, target_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'load_dataset' function implementation :-)")


# In[ ]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 2 (name: test1-2_load_dataset, points: 0.1)")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
target_col_name = 'imdb_score'

try:
    X, y = load_dataset(file_name, target_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert X.shape == (2956, 14), 'Wrong shape for feature vector dataframe'
assert y.shape[0] == 2956, 'Wrong number of target values in series'
np.testing.assert_array_equal(X.index, y.index, 'X and y should have the same index')

print ("Good Job!\nYou've passed the 2nd test for the 'load_dataset' function implementation :-)")


# In[ ]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 3 (name: test1-3_load_dataset, points: 0.1)")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
target_col_name = 'suitable_for_kids'
    
print ('Note: the dataset is already scaled ...')
try:
    X, y = load_dataset(file_name, target_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert X.shape == (2892, 15), 'Wrong shape for feature vector dataframe'
assert y.shape[0] == 2892, 'Wrong number of target values in series'

print ("Good Job!\nYou've passed the 3rd test for the 'load_dataset' function implementation :-)")


# ## 2. <u>The first</u> supervised model 
# In his section will we will do the following:
# * Train a supervised model on the train set
# * Predict the examples from the test set
# * Evaluate the performance of the model you've built

# In[ ]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: EXPLORATION

print ("The following are the first few rows of the 1st dataset:")
print ("-------------------------")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
target_col_name = 'imdb_score'

try:
    X, y = load_dataset(file_name, target_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

X.head()


# ### 2.a. Train <u>the first</u> supervised model 
# In this section you will train the first supervised model on the train set.
# 
# You need to train a model, which predicts the IMDb score.<br />
# * The value of the target series will be between 0 and 1.<br />
#   Higher values mean a better score.<br />
# * You could see the first few elements of the target series, from the output of the previous test.
# 
# You need to <u>identify the type of the learning problem</u>.<br />
# Accordingly, you need to understand whether to train a classification model or regression model.<br />
# * Train the supervised model, using the relevant `sklearn` algorithm you've studied: `LinearRegression` or `LogisticRegression`.

# ### 2.a. Instructions
# <u>method name</u>: <b>train_1st_model</b>
# <pre>The following is expected:
# --- Complete the 'train_1st_model' function to train a supervised learning model on the 
#     training set (use the input 'X_train' dataframe and the corresponding 'y_train' target values).
# You need to train a model, which predicts the IMDb score.
# 
# Note: you need to identify the learning problem and decide whether it is 
#           a classification or regression problem.
#       Accordingly, use the relevant sklearn algorithm for the training: 
#       LinearRegression or LogisticRegression.
# return the trained model.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return trained_model</b>

# In[ ]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def train_1st_model(X_train, y_train):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model


# In[ ]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'
X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 1 (name: test2a-1_train_1st_model, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'train_1st_model' ...")

file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'train_1st_model' function implementation :-)")


# In[ ]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 2 (name: test2a-2_train_1st_model, points: 0.3)")
print ("\t--->Testing the implementation of 'train_1st_model' ...")

file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert X_1st_train.shape == (2956, 14), 'Wrong shape for feature vector dataframe'
assert y_1st_train.shape[0] == 2956, 'Wrong number of target values in series'
np.testing.assert_array_equal(X_1st_train.index, y_1st_train.index, 'X and y should have the same index')

assert X_1st_test.shape == (739, 14), 'Wrong shape for feature vector dataframe'
assert y_1st_test.shape[0] == 739, 'Wrong number of target values in series'
np.testing.assert_array_equal(X_1st_test.index, y_1st_test.index, 'X and y should have the same index')
possible_python_classes = [sklearn.linear_model.LogisticRegression, sklearn.linear_model.LinearRegression]
assert type(trained_model_1st) in possible_python_classes, "Wrong retured type from 'train_1st_model' method, expected a 'LogisticRegression' or a 'LinearRegression' object"
    
print ("Good Job!\nYou've passed the 2nd test for the 'train_1st_model' function implementation :-)")


# In[ ]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 3 (name: test2a-3_train_1st_model, points: 0.2)")
print ("\t--->Testing the implementation of 'train_1st_model' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 2.b. Predict (IMDb score)
# In this section you will predict the IMDb score, for each of the examples in the test set,<br />
# using the first trained supervised model (based on your implementation in section 2.a.).

# ### 2.b. Instructions
# <u>method name</u>: <b>predict_1st</b>
# <pre>The following is expected:
# --- Complete the 'predict_1st' function, using the input 'trained_1st_model' model,
#     to predict the target of each example in the given 'X_test' test set.
# You need to return the predicted target values,
#    for each test set example.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return predicted_vals</b>

# In[ ]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def predict_1st(trained_1st_model, X_test):
    return trained_1st_model.predict(X_test)


# In[ ]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'
X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 1 (name: test2b-1_predict_1st, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'predict_1st' ...")
    
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'predict_1st' function implementation :-)")


# In[ ]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 2 (name: test2b-2_predict_1st, points: 0.3)")
print ("\t--->Testing the implementation of 'predict_1st' ...")
  
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
y_pred_1st= pd.Series(pred_1st_vals)
assert y_pred_1st.mean()!=y_1st_test.mean(), 'predicted values should not be identicle to actual test target values'
 
print ("Good Job!\nYou've passed the 2nd test for the 'predict_1st' function implementation :-)")


# In[ ]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 3 (name: test2b-3_predict_1st, points: 0.3)")
print ("\t--->Testing the implementation of 'predict_1st' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")
    
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 2.c. Evaluation of the first model (IMDb score prediction)
# In this section you will evaluate the performance of the first supervised model on the test set.
# 
# You will need to use an evaluation metric, which corresponds to the <u>type of learning problem</u>,
#    which you've identified in section 2.a.<br />
# 
# Accordingly, you need use one of the following relevant `sklearn` metrics you've studied: 
# * `r2_score` 
# * `f1_score`

# ### 2.c. Instructions
# <u>method name</u>: <b>evaluate_performance_1st</b>
# <pre>The following is expected:
# --- Complete the 'evaluate_performance_1st' function to evaluate the performance of the
#       1st supervised learning model (IMDb score prediction), on the test set examples.
#    Accordingly, you need use either the 'r2_score' or the 'f1_score' evaluation metric.
# The input parameters:
#                      y_test - a series of containing all actual target values per test instance
#                      y_predicted -  a series which contains all the predicted values per test instance 
#                                    (using the 1st model).
# Note: You need to return the result of the evaluation metric (floating number).
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return evaluate_value</b>

# In[ ]:


# 2.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def evaluate_performance_1st(y_test,y_predicted):
    return r2_score(y_test,y_predicted)


# In[ ]:


# 2.c
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'  
X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 2.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 2.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.c. - Test 1 (name: test2c-1_evaluate_performance_1st, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'evaluate_performance_1st' ...")

file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'

try:    
    X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise
    
print ("Good Job!\nYou've passed the 1st test for the 'evaluate_performance_1st' function implementation :-)")


# In[ ]:


# 2.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.c. - Test 2 (name: test2c-2_evaluate_performance_1st, points: 0.4)")
print ("\t--->Testing the implementation of 'evaluate_performance_1st' ...")
    
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'imdb_score'

try:    
    X_1st_train, y_1st_train = load_dataset(file_name_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_name_test, target_col_name)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert np.round(eval_res_1st+0.001,2) == 0.34, 'Wrong evaluation result'

print ("Good Job!\nYou've passed the 2nd test for the 'evaluate_performance_1st' function implementation :-)")


# ## 3. <u>The second</u> supervised model 
# In his section will we will do the following:
# * Train a supervised model on the train set
# * Predict the examples from the test set
# * Evaluate the performance of the model you've built

# In[ ]:


# 3.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: EXPLORATION

print ("The following are the first few rows of the 2nd dataset:")
print ("-------------------------")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
target_col_name = 'suitable_for_kids'
    
print ('Note: the dataset is already scaled ...')

try:
    X, y = load_dataset(file_name, target_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

X.head()


# ### 3.a. Train <u>the second</u> supervised model 
# In this section you will train the second supervised model on the train set.
# 
# You need to train a model, which predicts whether the movie is suitable for children.<br />
# * Expect a value of 1 in the target series, if the movie is suitable for children.<br/>
#   Otherwise, expect a value of 0.<br />
# * You could see the first few elements of the target series, from the output of the previous test.
# 
# You need to <u>identify the type of the learning problem</u>.<br />
# Accordingly, you need to understand whether to train a classification model or regression model.<br />
# * Train the supervised model, using the relevant `sklearn` algorithm you've studied: `LinearRegression` or `LogisticRegression`.

# ### 3.a. Instructions
# <u>method name</u>: <b>train_2nd_model</b>
# <pre>The following is expected:
# --- Complete the 'train_2nd_model' function to train a supervised learning model on the 
#       training set (use the give 'X_train' dataframe and the corresponding 'y_train' target values).
# You need to train a model, which predicts whether the movie is suitable for children.
#    Identify the learning problem and decide whether it is a classification or regression problem.
#       Accordingly, use the relevant sklearn algorithm for the training: 
#       LinearRegression or LogisticRegression.
# return the trained model.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return trained_model</b>

# In[ ]:


### 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def train_2nd_model(X_train, y_train):
    return LogisticRegression(random_state=0).fit(X_train, y_train)



# In[ ]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'
X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 1 (name: test3a-1_train_2nd_model, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'train_2nd_model' ...")
    
file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'

try:    
    X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
    X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
    trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'train_2nd_model' function implementation :-)")


# In[ ]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 2 (name: test3a-2_train_2nd_model, points: 0.3)")
print ("\t--->Testing the implementation of 'train_2nd_model' ...")

file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'

try:    
    X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
    X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
    trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert X_2nd_train.shape == (2892, 15), 'Wrong shape for feature vector dataframe'
assert y_2nd_train.shape[0] == 2892, 'Wrong number of target values in series'
np.testing.assert_array_equal(X_2nd_train.index, y_2nd_train.index, 'X and y should have the same index')

assert X_2nd_test.shape == (724, 15), 'Wrong shape for feature vector dataframe'
assert y_2nd_test.shape[0] == 724, 'Wrong number of target values in series'
np.testing.assert_array_equal(X_2nd_test.index, y_2nd_test.index, 'X and y should have the same index')
possible_python_classes = [sklearn.linear_model.LogisticRegression, sklearn.linear_model.LinearRegression]
assert type(trained_model_2nd) in possible_python_classes, "Wrong retured type from 'train_2nd_model' method, expected a 'LogisticRegression' or a 'LinearRegression' object"
    
print ("Good Job!\nYou've passed the 2nd test for the 'train_2nd_model' function implementation :-)")


# In[ ]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 
print ("Part 3.a. - Test 3 (name: test3a-3_train_2nd_model, points: 0.2)")
print ("\t--->Testing the implementation of 'train_2nd_model' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 3.b. Predict (is movie suitable for children)
# In this section you will predict whether the movies are suitable for children, for each of the <br />
# examples in the test set, using the second trained supervised model (based on your implementation in 3.a.).

# ### 3.b. Instructions
# <u>method name</u>: <b>predict_2nd</b>
# <pre>The following is expected:
# --- Complete the 'predict_2nd' function, using the input 'trained_2nd_model' model,
#       to predict the target of each example in the given 'X_test' test set.
# You need to return the predicted target values,
#    for each test set example.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return predicted_vals</b>

# In[ ]:


### 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def predict_2nd(trained_2nd_model, X_test):
    return trained_2nd_model.predict(X_test)


# In[ ]:


# 3.b.
# ---->>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'
X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
pred_2nd_vals = predict_2nd(trained_model_2nd, X_2nd_test)
y_pred_2nd = pd.Series(pred_2nd_vals,index=X_2nd_test.index)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 1 (name: test3b-1_predict_2nd, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'predict_2nd' ...")

file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'

try:    
    X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
    X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
    trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
    pred_2nd_vals = predict_2nd(trained_model_2nd, X_2nd_test)
    y_pred_2nd = pd.Series(pred_2nd_vals,index=X_2nd_test.index)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'predict_2nd' function implementation :-)")


# In[ ]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 2 (name: test3b-2_predict_2nd, points: 0.3)")
print ("\t--->Testing the implementation of 'predict_2nd' ...")

file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'

try:    
    X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
    X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
    trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
    pred_2nd_vals = predict_2nd(trained_model_2nd, X_2nd_test)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

y_pred_2nd = pd.Series(pred_2nd_vals,index=X_2nd_test.index)
assert y_pred_2nd.mean()!=y_2nd_test.mean(), 'predicted values should not be identicle to actual test target values'
     
print ("Good Job!\nYou've passed the 2nd test for the 'predict_2nd' function implementation :-)")


# In[ ]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 3 (name: test3b-3_predict_2nd, points: 0.2)")
print ("\t--->Testing the implementation of 'predict_2nd' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 3.c. Evaluation of the second model ('is movie suitable for children' prediction)
# In this section you will evaluate the performance of the second supervised model on the test set.
# 
# You will need to use an evaluation metric, which corresponds to the <u>type of learning problem</u>,
#    which you've identified in section 3.a.<br />
# 
# Accordingly, you need use one of the following relevant `sklearn` metrics you've studied: 
# * `r2_score` 
# * `f1_score`

# ### 3.c. Instructions
# <u>method name</u>: <b>evaluate_performance_2nd</b>
# <pre>The following is expected:
# --- Complete the 'evaluate_performance_2nd' function to evaluate the performance of the
#       2nd supervised learning model ('is movie suitable for children' prediction), 
#       on the test set examples.
#    Accordingly, you need use either the 'r2_score' or the 'f1_score' evaluation metric.
# The input parameters:
#                      y_test - a series of containing all actual target values per test instance
#                      y_predicted -  a series which contains all the predicted values per test instance 
#                                    (using the 2nd model).
# You need to return the result of the evaluation metric (floating number).
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return evaluate_value</b>

# In[ ]:


### 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def evaluate_performance_2nd(y_test,y_predicted):
    return metrics.f1_score(y_test, y_predicted)


# In[ ]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'
X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
pred_2nd_vals = predict_2nd(trained_model_2nd, X_2nd_test)
y_pred_2nd = pd.Series(pred_2nd_vals,index=X_2nd_test.index)
eval_res_2nd = evaluate_performance_2nd(y_2nd_test, y_pred_2nd)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.c. - Test 1 (name: test3c-1_evaluate_performance_2nd, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'evaluate_performance_2nd' ...")

file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'

try:    
    X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
    X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
    trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
    pred_2nd_vals = predict_2nd(trained_model_2nd, X_2nd_test)
    y_pred_2nd = pd.Series(pred_2nd_vals,index=X_2nd_test.index)
    eval_res_2nd = evaluate_performance_2nd(y_2nd_test, y_pred_2nd)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

print ("Good Job!\nYou've passed the 1st test for the 'evaluate_performance_2nd' function implementation :-)")


# In[ ]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.c. - Test 2 (name: test3c-2_evaluate_performance_2nd, points: 0.4)")
print ("\t--->Testing the implementation of 'evaluate_performance_2nd' ...")

file_name_2nd_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_train.csv'
file_name_2nd_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_2nd_model_test.csv'
target_col_2nd_name = 'suitable_for_kids'

try:    
    X_2nd_train, y_2nd_train = load_dataset(file_name_2nd_train, target_col_2nd_name)
    X_2nd_test, y_2nd_test = load_dataset(file_name_2nd_test, target_col_2nd_name)
    trained_model_2nd = train_2nd_model(X_2nd_train, y_2nd_train)
    pred_2nd_vals = predict_2nd(trained_model_2nd, X_2nd_test)
    y_pred_2nd = pd.Series(pred_2nd_vals,index=X_2nd_test.index)
    eval_res_2nd = evaluate_performance_2nd(y_2nd_test, y_pred_2nd)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise
# ---------------------------
assert np.round(eval_res_2nd+0.001,2)  == 0.69, 'Wrong evaluation result'

print ("Good Job!\nYou've passed the 2nd test for the 'evaluate_performance_2nd' function implementation :-)")


# ## 4. Feature manipulation  
# In this section you need to perform feature manipulation, in order to<br />
# improve the performance of the 1st supervised model.<br />
# 
# ### Important note:
# The improvements should be consistent with your identification of the learning problem in section 2.a.
# 
# Accordigly we will use the following functions for testing:
# * train_1st_model - to train the model with your improvements
# * predict_1st     - to predict the test examples
# * evaluate_performance_1st -to evaluate the performance of the model after the feature manipulation
#   improvements.
# 
# ### About the tests in this section:
# As mentioned above, the tests of this sections, will use the 'evaluate_performance_1st' to evaluate the permofmace of the 
# model after the improvements.<br />
# The tests will compare the performance with the performance of the trained model, without the improvements.<br />
# Using the evaluation metic (implemeted in 'evaluate_performance_1st') the expected difference is as following:<br />
# 
# `Eval_improved` - evaluation performance of the <u>improved trained model</u> (model with your improvements).<br/>
# `Eval_basic`    -  evaluation performance of the <u>basic trained model</u>  (model without improvements)<br />
# <u> The criteria to pass the test will check what is the value of `Eval_improved` - `Eval_basic` </u>
# * First 2 tests, are basic syntax test and will give you 0.3 points if you pass them.
# * If  `Eval_improved` - `Eval_basic` > 0, you pass the 1st test     (total 0.7 a point if you pass this test)
# * If  `Eval_improved` - `Eval_basic` > 0.015, you pass the 2nd test (total 1.1 point if you pass this test too)
# * If  `Eval_improved` - `Eval_basic` > 0.03, you pass the 3rd test  (total 1.6 points if you pass this test too)
# * If  `Eval_improved` - `Eval_basic` > 0.04, you pass the 3rd test  (total 2.1 points if you pass this test too)
# 
# ### About the dataset
# The Dataset includes a few additional features, which were not included in the original dataset.<br />
# Note: the <u>new features that were added all contain string values</u>. If you decide to use these features, they <b>must to be transfered to numerical features.</b><br />
# The following list is an explaination of these new features (all the features contain string values):<br />
# * 'color' - It indicates whether the movie is in Color or Black & White
# * 'language' - The main spoken language in the movie, such as English, Dutch, Mandarin and so on.
# * 'content_rating' - The group category for which the movie is appropriate for, such as 'PG-13', 'PG', 'R' and so on.
# * 'country' - The country which the film took place in.
# 
# The following section includes the first few lines of this dataset:

# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: EXPLORATION

print ("The first few lines of this dataset:")
print ("-----------------------------------")

file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
target_col_name = 'imdb_score'
    
print ('Note: the dataset is already scaled ...')

try:
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

X_1st_manip_train.head()


# ### Imprtant notes
# <div class="alert alert-danger">
# &#x2718; <u>do not change the  values of the target</u><br />
# &#x2718; <u>do not leave features, which contain string values after this process</u>
# </div>
# 
# ### Perform feature manipulation
# You could perform one or more of the following manipulations, or any other manipulation you've studies, which you could think of:
# <div class="alert alert-success">
# <ul>
#   <li>Transfer the values of the new features from string to numerical values, For instance:
#     <ul>
#       <li>In some 'col1' feature: 'str_val_1' --> 0</li>
#       <li>'str_val_2' -->1</li>
#       <li>...</li>
#     </ul>
#   </li>
#   <li>Use the methods you studies to group values and transfer them to categorical numeric values, For instance:
#     <ul><li>Group numerical values into bins of into binary values.</li></ul>
#   </li>
#   <li>Performe mathimatical comninations between two features, such as:
#     <ul>
#         <li>df['new_feature'] = df['col_name1']-df['col_name2']</li>
#         <li>df['new_feature'] = df['col_name1']/df['col_name2'] # make sure you don't divide in zero</li>
#      </ul>
#   <li>Remove features, which might not be helpful.</li>
# </ul><br />
# <u>For instance</u>, we recommend that you try unifying countries and features regarding Facebook likes.
# </div>

# ### 4. Instructions
# <u>method name</u>: <b>manipulate_1st_feature_vector</b>
# <pre>The following is expected:
# --- Complete the 'manipulate_1st_feature_vector' function to return a dataframe, for which
#      the evaluation of the 1st learning problem should improve, using feature manipulation.
# This could include any manipulation of the features, which you've studied, removal of features,
#    and also creation of new features, which contain some sort of combination
#    of the original features (such as a division of the value of two other features and so on).
# Notes:
#       Features which are not numeric should be removed from the output (otherwise, the training will fail).
#       Again the improvements should be consistent with your identification of the learning problem in section 2.a.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return processed_X</b>

# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[ ]:


### 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def manipulate_1st_feature_vector(X):

    X['color']=X['color'].replace('Color',1)
    X['color']=X['color'].replace('Black and White',0)

    lang_arr=[i for i in  X['language'].unique()]
    
    X['language'].replace('English',1,inplace=True)
    X['language'].replace(lang_arr,0,inplace=True)


    content_arr=[i for i in  X['content_rating'].unique()]
    j=0
    for i in content_arr:
        X['content_rating'].replace([i],j,inplace=True)
        j+=1

    X['num_user_for_reviews'].replace(0,1,inplace=True)
    X['avg_user_rev'] = X['num_voted_users']/X['num_user_for_reviews']
    X['earnings']=X['budget']/X['gross']
    X['num_critic_for_reviews'].replace(0,1,inplace=True)
    X['avg like'] = X['movie_facebook_likes']/X['num_critic_for_reviews']
    X.drop(['country','facenumber_in_poster','color'], axis=1, inplace=True)

    return X


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:file_basic_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'
X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 1 (name: test4-1_manipulate_1st_feature_vector, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'manipulate_1st_feature_vector' ...")

file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
    X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
    X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
    X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

print ("Good Job!\nYou've passed the 1st test for the 'manipulate_1st_feature_vector' function implementation :-)")


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 2 (name: test4-2_manipulate_1st_feature_vector, points: 0.2)")
print ("\t--->Testing the implementation of 'manipulate_1st_feature_vector' ...")

file_basic_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_basic_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_train, y_1st_train = load_dataset(file_basic_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_basic_test, target_col_name)
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
    X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
    X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
    X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    trained_model_manip_1st = train_1st_model(X_1st_manip_train, y_1st_manip_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    pred_manip_1st_vals = predict_1st(trained_model_manip_1st, X_1st_manip_test)
    y_pred_manip_1st = pd.Series(pred_manip_1st_vals,index=X_1st_manip_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
    eval_res_manip_1st = evaluate_performance_1st(y_1st_manip_test, y_pred_manip_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert X_1st_train.shape == (2956, 14), 'Wrong shape for feature vector dataframe'
assert X_1st_test.shape == (739, 14), 'Wrong shape for feature vector dataframe'
np.testing.assert_array_equal(X_1st_train.index, y_1st_train.index, 'X and y should have the same index')
np.testing.assert_array_equal(X_1st_manip_train.index, X_1st_train.index, "index shouldn't change after feature manipulation")
np.testing.assert_array_equal(X_1st_manip_test.index, X_1st_test.index, "index shouldn't change after feature manipulation")
np.testing.assert_array_equal(X_1st_test.index, y_1st_test.index, 'X and y should have the same index')
np.testing.assert_array_equal(X_1st_manip_test.index, y_1st_manip_test.index, 'X and y should have the same index')
np.testing.assert_array_equal(y_1st_manip_test.index, y_pred_manip_1st.index, 'pred and test should have the same index')

print ("Good Job!\nYou've passed the 2nd test for the 'manipulate_1st_feature_vector' function implementation :-)")


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 3 (name: test4-3_manipulate_1st_feature_vector, points: 0.4)")
print ("\t--->Testing the implementation of 'manipulate_1st_feature_vector' ...")
print ("\n\t====> model improvement test - in order to get the points for this test,")
print ("\t                               you need to improve the performance of the basic model,")
print ("\t                               using your implementation in the ")
print ("\t                               'manipulate_1st_feature_vector' function")

file_basic_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_basic_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'

try:
    X_1st_train, y_1st_train = load_dataset(file_basic_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_basic_test, target_col_name)
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
    X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
    X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
    X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    trained_model_manip_1st = train_1st_model(X_1st_manip_train, y_1st_manip_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    pred_manip_1st_vals = predict_1st(trained_model_manip_1st, X_1st_manip_test)
    y_pred_manip_1st = pd.Series(pred_manip_1st_vals,index=X_1st_manip_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
    eval_res_manip_1st = evaluate_performance_1st(y_1st_manip_test, y_pred_manip_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise
    
assert np.round(eval_res_manip_1st-eval_res_1st,3) > 0, 'improved model must have a better performance to pass this test ...'

print ("Good Job!\nYou've passed the 3rd test for the 'manipulate_1st_feature_vector' function implementation :-)")


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 4 (name: test4-4_manipulate_1st_feature_vector, points: 0.5)")
print ("\t--->Testing the implementation of 'manipulate_1st_feature_vector' ...")
print ("\n\t====> model improvement test - in order to get the points for this test,")
print ("\t                               you need to improve the performance of the basic model")
print ("\t                               in MORE THAN 0.015, using your implementation in the ")
print ("\t                               'manipulate_1st_feature_vector' function")

file_basic_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_basic_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'

try:    
    X_1st_train, y_1st_train = load_dataset(file_basic_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_basic_test, target_col_name)
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
    X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
    X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
    X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    trained_model_manip_1st = train_1st_model(X_1st_manip_train, y_1st_manip_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    pred_manip_1st_vals = predict_1st(trained_model_manip_1st, X_1st_manip_test)
    y_pred_manip_1st = pd.Series(pred_manip_1st_vals,index=X_1st_manip_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
    eval_res_manip_1st = evaluate_performance_1st(y_1st_manip_test, y_pred_manip_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert np.round(eval_res_manip_1st-eval_res_1st,4) > 0.015, 'improvement was less than  0.015, not enough for this test'

print ("Good Job!\nYou've passed the 4th test for the 'manipulate_1st_feature_vector' function implementation :-)")


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 5 (name: test4-5_manipulate_1st_feature_vector, points: 0.5)")
print ("\t--->Testing the implementation of 'manipulate_1st_feature_vector' ...")
print ("\n\t====> model improvement test - in order to get the points for this test,")
print ("\t                               you need to improve the performance of the basic model")
print ("\t                               in MORE THAN 0.03, using your implementation in the ")
print ("\t                               'manipulate_1st_feature_vector' function")

file_basic_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_basic_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'

try:    
    X_1st_train, y_1st_train = load_dataset(file_basic_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_basic_test, target_col_name)
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
    X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
    X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
    X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    trained_model_manip_1st = train_1st_model(X_1st_manip_train, y_1st_manip_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    pred_manip_1st_vals = predict_1st(trained_model_manip_1st, X_1st_manip_test)
    y_pred_manip_1st = pd.Series(pred_manip_1st_vals,index=X_1st_manip_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
    eval_res_manip_1st = evaluate_performance_1st(y_1st_manip_test, y_pred_manip_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert np.round(eval_res_manip_1st-eval_res_1st,3) > 0.03, 'improvement was less than 0.03, not enough for this test'

print ("Good Job!\nYou've passed the 5th test for the 'manipulate_1st_feature_vector' function implementation :-)")


# In[ ]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 6 (name: test4-6_manipulate_1st_feature_vector, points: 0.5)")
print ("\t--->Testing the implementation of 'manipulate_1st_feature_vector' ...")
print ("\n\t====> model improvement test - in order to get the points for this test,")
print ("\t                               you need to improve the performance of the basic model")
print ("\t                               in MORE THAN 0.04, using your implementation in the ")
print ("\t                               'manipulate_1st_feature_vector' function")

file_basic_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
file_basic_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_train.csv'
file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_w_cat_test.csv'
target_col_name = 'imdb_score'

try:    
    X_1st_train, y_1st_train = load_dataset(file_basic_train, target_col_name)
    X_1st_test, y_1st_test = load_dataset(file_basic_test, target_col_name)
    X_1st_manip_train, y_1st_manip_train = load_dataset(file_name_train, target_col_name)
    X_1st_manip_test, y_1st_manip_test = load_dataset(file_name_test, target_col_name)
    X_1st_manip_train = manipulate_1st_feature_vector(X_1st_manip_train)
    X_1st_manip_test = manipulate_1st_feature_vector(X_1st_manip_test)
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    trained_model_manip_1st = train_1st_model(X_1st_manip_train, y_1st_manip_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    pred_manip_1st_vals = predict_1st(trained_model_manip_1st, X_1st_manip_test)
    y_pred_manip_1st = pd.Series(pred_manip_1st_vals,index=X_1st_manip_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
    eval_res_manip_1st = evaluate_performance_1st(y_1st_manip_test, y_pred_manip_1st)
except Exception as e:
    print ('You probably have a syntax or implementation error,  \nerror Message:',str(e), '\nTry fixing your code')
    raise

assert np.round(eval_res_manip_1st-eval_res_1st,3) > 0.04, 'improvement not enough for this test'

print ("Good Job!\nYou've passed the 6th test for the 'manipulate_1st_feature_vector' function implementation :-)")

