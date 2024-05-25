#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd


# In[2]:


# Load the dataset

data = pd.read_csv("aircraft_engine_maintenance_supervised_learning.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


# Check missing values using msno

import missingno as msno

msno.bar(data, color='grey')


# In[8]:


# Display columns with missing values
import seaborn as sns

sns.heatmap(data.isnull())


# In[9]:


# Impute missing values (replace with median) on numerical columns

numerical_columns = ["Engine_ID", "Temperature", "Pressure", "Rotational_Speed", "Engine_Health",
                    'Fuel_Consumption', 'Vibration_Level', 'Oil_Temperature', 'Altitude',
                     'Humidity', 'Maintenance_Needed']


data.fillna(data[numerical_columns].median(), inplace=True)


# In[10]:


numerical_columns


# In[11]:


data.isnull().sum()


# In[12]:


# Check missing values using msno after filling with median value

import missingno as msno

msno.bar(data, color='grey')


# ### Exploratory Data Analysis
# 

# In[13]:


# plotting boxplot to visualise outliers
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Univariate analysis (Numerical columns) - use boxplot to view all numerical columns with outliers.
columns_to_check = ['Temperature','Pressure','Rotational_Speed','Engine_Health','Fuel_Consumption',
 'Vibration_Level','Oil_Temperature','Altitude','Humidity']


plt.figure(figsize=(18, 10))
for i, cols in enumerate(columns_to_check, 1):

    plt.subplot(2, 5, i)
    sns.boxplot(data[cols])
    plt.xlabel(cols)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ### Univariate analysis

# In[14]:


# Univariate analysis (Target column i.e. Maintenance_Needed) - Countplot

plt.figure(figsize=(6,4))
sns.countplot(data=data, x="Maintenance_Needed")


# ### Bivariate analysis

# In[15]:


# Bivariate analysis - Pairplot

sns.pairplot(data[columns_to_check + ["Maintenance_Needed", "Remaining_used_life"]], hue="Maintenance_Needed")
plt.show()


# ### Multivariate analysis

# In[16]:


# Multivariate analysis - Correlation Heatmap

correlation_matrix = data[columns_to_check + ["Maintenance_Needed", "Remaining_used_life"]].corr()
correlation_matrix


# In[17]:


# correllation heatmap showing levl of correlation in the features
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")


# ###  Feature Engineering:
# #### Create a Time-Related Feature
# #### Binning 'Hour_of_Day'
# 
# 

# ### Data Pre-processing

# In[18]:


# Create a Time-Related Feature (hour_of_Day)
dataCopy = data.copy()

dataCopy['Timestamp'] = pd.to_datetime(dataCopy['Timestamp'], errors='coerce')

dataCopy = dataCopy.dropna(subset=['Timestamp'])
dataCopy['Hour_of_Day'] = dataCopy['Timestamp'].dt.hour

#display(data.dtypes)

dataCopy.head()


# In[19]:


# Binning 'Hour_of_Day'
bins = [-1, 6, 12, 18, 24]
labels = ['Night', 'Morning', 'Afternoon', 'Evening']
dataCopy['Day_Part'] = pd.cut(dataCopy['Hour_of_Day'], bins=bins, labels=labels, right=False)

# Display the dataset with the new feature
dataCopy.head()


# #### Feature Scaling a.k.a outlier handler:

# In[20]:


# Feature selections
labels = data[['Maintenance_Needed']]
features = data.drop(['Maintenance_Needed','Unnamed: 0','Engine_ID','Timestamp', 'Remaining_used_life'], axis=1)


# In[21]:


features.head()


# In[22]:


print(labels.shape)
print(labels.squeeze().shape)


# In[23]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels.squeeze(), test_size=0.2, random_state=0)

# Standardize our training data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['number']))

X_test_scaled = scaler.transform(X_test.select_dtypes(include=['number']))


# In[ ]:





# ### Machine Learning Modelling

# In[24]:


X_train_scaled.shape


# In[25]:


X_test_scaled.shape


# In[26]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[27]:


# Initialize the model

log_reg_model = LogisticRegression(random_state=0)


# In[28]:


# Train the model
log_reg_model.fit(X_train_scaled, y_train)


# In[29]:


# Make predictions on the test set
y_pred = log_reg_model.predict(X_test_scaled)


# In[30]:


y_pred


# In[31]:


# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)


# In[32]:


# Display results
print("Accuracy: ", accuracy*100)
print("Classification Report: \n", report)
print("Confusion Matrix: \n", matrix)


# In[33]:


cm = confusion_matrix(y_test,y_pred)

ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')

plt.xlabel("Predicted Class",fontsize=12) 
plt.ylabel("True Class",fontsize=12) 
plt.title("Confusion Matrix",fontsize=12)

plt.show()


# ### Is this the best we can achieve ? Why not try different classification models?

# In[34]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize and train the models
models = {
    "Logistic Regression" : LogisticRegression(random_state=0),
    "Decision Tree" : DecisionTreeClassifier(random_state=0),
    "Random Forest" : RandomForestClassifier(random_state=0),
    "SVC" : SVC(random_state=0)
         }


# In[35]:


for key, val in models.items():
    print(key, "=", val)


# In[36]:


for model_name, model in models.items():
    
    # Training and prediction
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Display results
    print(model_name)
    print("Accuracy: ", accuracy*100)
    print("Classification Report: \n", report)
    plt.figure(figsize=(5,3))
    #sns.heatmap(matrix, annot=True)
    sns.heatmap(matrix, cmap='flare',annot=True, fmt='d')
    plt.xlabel("Predicted Class",fontsize=10) 
    plt.ylabel("True Class",fontsize=10) 
    plt.title("Confusion Matrix",fontsize=10)
    plt.show()
    print("\n")


# ### Can we do better ? Why don't we select the most important features and train with them ?

# In[37]:


import matplotlib.pyplot as plt

# Initialize and train the Random Forest model
radom_forest_model = RandomForestClassifier(random_state=0)
radom_forest_model.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances =  radom_forest_model.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importances_df = pd.DataFrame({'Features': X_train.columns, 'Importance': feature_importances})

# Sort features by importance in descending order
feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot the feature importance
sns.barplot(x='Importance', y='Features', data=feature_importances_df, palette='viridis')


# #### Let's select the top 5 features

# In[38]:


selected_features = ['Engine_Health', 'Vibration_Level', 'Rotational_Speed', 'Oil_Temperature']


# In[39]:


# Split the data into training and testing sets

X_train_, X_test_, y_train_, y_test_ = train_test_split(data[selected_features], labels.squeeze(), test_size=0.2, 
                                                        random_state=0)

# Standardize our training data
scaler = StandardScaler()

X_train_scaled_ = scaler.fit_transform(X_train_.select_dtypes(include=['number']))

X_test_scaled_ = scaler.transform(X_test_.select_dtypes(include=['number']))


# In[40]:


# Train, Test, and Evaluate Model 
for model_name, model in models.items():
    
    # Training and prediction
    model.fit(X_train_scaled_, y_train_)
    y_pred = model.predict(X_test_scaled_)

    # Evaluate the model
    accuracy = accuracy_score(y_test_, y_pred)
    report = classification_report(y_test_, y_pred)
    matrix = confusion_matrix(y_test_, y_pred)

    # Display results
    print(model_name)
    print("Accuracy: ", accuracy*100)
    print("Classification Report: \n", report)
    plt.figure(figsize=(4,2))
    sns.heatmap(matrix, cmap='flare',annot=True, fmt='d')
    plt.xlabel("Predicted Class",fontsize=10) 
    plt.ylabel("True Class",fontsize=10) 
    plt.title("Confusion Matrix",fontsize=10)
    plt.show()
    print("\n")


# ## Hyper-parameter Tunning
# ### How can I automate this process of selecting the best parameters to train my model ?

# In[41]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth' : [None, 10, 20],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
radom_forest_model = RandomForestClassifier(random_state=0)

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=radom_forest_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled_, y_train_)

# Get the best hyperparameters
best_param = grid_search.best_params_
best_param


# In[42]:


# Train the model with the best hyperparameters
best_radom_forest_model = RandomForestClassifier(random_state=0, **best_param)
best_radom_forest_model.fit(X_train_scaled_, y_train_)

# Make predictions on the test set
y_pred = best_radom_forest_model.predict(X_test_scaled_)

# Evaluate the model
accuracy = accuracy_score(y_test_, y_pred)
report = classification_report(y_test_, y_pred)
matrix = confusion_matrix(y_test_, y_pred)

# Display results
print(model_name)
print("Accuracy: ", accuracy*100)
print("Classification Report: \n", report)
plt.figure(figsize=(4,2))
sns.heatmap(matrix, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class",fontsize=10) 
plt.ylabel("True Class",fontsize=10) 
plt.title("Confusion Matrix",fontsize=10)
plt.show()
print("\n")


# ### Regression Analysis

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize linear regression model
lin_reg_model = LinearRegression()

labels = data[['Remaining_used_life']]
features = data.drop(['Maintenance_Needed','Unnamed: 0','Engine_ID','Timestamp', 'Remaining_used_life'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels.squeeze(), test_size=0.2, random_state=0)

# Standardize our training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['number']))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=['number']))

# Training
lin_reg_model.fit(X_train_scaled, y_train)


# Prediction on the training and testing set
y_train_pred = lin_reg_model.predict(X_train_scaled)
y_test_pred = lin_reg_model.predict(X_test_scaled)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

train_r2_score = r2_score(y_train, y_train_pred)
test_r2_score = r2_score(y_test, y_test_pred)

print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)

print("Train R^2: ", train_r2_score)
print("Test R^2: ", test_r2_score)

