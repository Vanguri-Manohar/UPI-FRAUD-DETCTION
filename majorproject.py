#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PS_20174392719_1491204439457_log.csv
# For Data Analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Fraud_D = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Remove the last column
Fraud_D = Fraud_D.iloc[:, :-1]


# In[3]:


Fraud_D.columns= ["step", "type", "amount", "customer_starting_transaction", "bal_before_transaction", 
            "bal_after_transaction", "recipient_of_transaction", "bal_of_recepient_before_transaction", "bal_of_receipient_after_transaction", "fraud_transaction"]


# In[4]:


# View data (to give you first five rows)
Fraud_D.head()


# In[5]:


# View data (to give you last five rows)
Fraud_D.tail()  


# In[6]:


#Data Verification

Fraud_D.info()


# In[7]:


# statistical analysis of the data

Fraud_D.describe()


# In[8]:


Fraud_D.describe().astype(int)


# In[9]:


#Missing values

Fraud_D.isnull()


# In[10]:


Fraud_D.isnull().sum()


# In[11]:


# To visualize the missing values

plt.figure(figsize = (10,5))
plt.title ("missing data visualization in the dataset")
sns.heatmap(Fraud_D.isnull(), cbar =True, cmap= "Blues_r")


# In[12]:


#check shape of the entire dataframe using .shape attribute
Fraud_D.shape


# In[13]:


# We have 6362620 rows and 10 columns in the dataset
# EXPLORATORY DATA ANALYSIS
# Univariate Analysis

# Bivariate Analysis

# Multivariate Analysis

# Correlation


# In[14]:


# Univariate Analysis
#visualize type of online transaction
plt.figure(figsize=(10,5))
sns.countplot (x="type", data= Fraud_D)
plt.title ("Visualizing type of online transaction")
plt.xlabel("Type of online transaction")
plt.ylabel("count of online transaction type ")


# In[15]:


# From the chart, it is seen that cash_out and payment is the most common type of online transaction that customers use


# In[16]:


# create a function that properly labels isFraud

def Fraud (x):
    if x ==1:
        return "Fraudulent"
    else:
        return "not Fraudulent"
    
# create a new column
Fraud_D["fraud_transaction_label"] = Fraud_D["fraud_transaction"].apply(Fraud)


# create visualization
plt.figure(figsize = (10,5))
plt.title ("Fraudulent Transactions")
Fraud_D.fraud_transaction_label.value_counts().plot.pie(autopct='%1.1f%%')


# In[17]:


# From this chart, its shows that most of the online transactions customers does is not fraudulent. Also the dataset is not balance


# In[18]:


Fraud_D.fraud_transaction_label.value_counts()


# In[19]:


8213/6354407*100


# In[20]:


# 8,213 transactions have been tagged as fraudulent in the dataset, which is approximately 13% of the total number of transactions.


# In[21]:


#To disable warnings
import warnings
warnings.filterwarnings("ignore")

# Visualization for step column

plt.figure(figsize=(15,6))
sns.distplot(Fraud_D['step'],bins=100)


# In[22]:


# The above graph indicates the distribution of the step column


# In[23]:


# Visualization for amount column

sns.histplot(x= "amount", data =Fraud_D)


# In[24]:


Fraud_D.head()


# In[25]:


Fraud_D.tail()


# In[26]:


# Bivariate Analysis

sns.barplot(x='type',y='amount',data=Fraud_D,ci=None)


# In[27]:


# In this chart, 'transfer' type has the maximum amount of money being transfered from customers to the recipient. Although 'cash out' and 'cash_in 'are the most common type of transactions


# In[28]:


# Visualization between step and amount

sns.jointplot(x='step',y='amount',data=Fraud_D)


# In[29]:


sns.scatterplot(x=Fraud_D["amount"], y=Fraud_D["step"])


# In[30]:


# Visualization between amount and fraud_transaction_label

plt.figure(figsize=(15,6))
plt.scatter(x='amount',y='fraud_transaction_label',data=Fraud_D)
plt.xlabel('amount')
plt.ylabel('fraud_transaction_label')


# In[31]:


# Although the amount of fraudulent transactions is very low, majority of them are constituted within 0 and 10,000,000 amount.


# In[32]:


# Visualization between type and isfraud_label

plt.scatter(x='type',y='fraud_transaction_label',data=Fraud_D)
plt.xlabel('type')
plt.ylabel('fraud_transaction_label')


# In[33]:


# Visualization between type and isfraud_label

plt.figure(figsize=(12,8))
sns.countplot(x='fraud_transaction_label',data=Fraud_D,hue='type')
plt.legend(loc=[0.85,0.8])


# In[34]:


# Both the above graphs indicate that transactions of the type 'transfer' and 'cash out' comprise fraudulent transactions


# In[35]:


# Multivariate Analysis

# Visualizing btw step,type and isFraud_label

sns.boxplot(x= "type", y= "step", hue ="fraud_transaction_label", data= Fraud_D)


# In[36]:


# Correlation

corel= Fraud_D.corr()
sns.heatmap(corel, annot =True)


# In[37]:


# One Hot Encoding
#1. select categorical variables

categorical = ['type']


# In[38]:


#2. use pd.get_dummies() for one hot encoding
#replace pass with your code

categories_dummies = pd.get_dummies(Fraud_D[categorical])

#view what you have done
categories_dummies.head()


# In[39]:


#join the encoded variables back to the main dataframe using pd.concat()
#pass both data and categories_dummies as a list of their names
#pop out documentation for pd.concat() to clarify

Fraud_D = pd.concat([Fraud_D,categories_dummies], axis=1)

#check what you have done
print(Fraud_D.shape)
Fraud_D.head()


# In[40]:


#remove the initial categorical columns now that we have encoded them
#use the list called categorical to delete all the initially selected columns at once

Fraud_D.drop(categorical, axis = 1, inplace = True)

Fraud_D.drop(columns=['fraud_transaction_label', 'customer_starting_transaction', 'recipient_of_transaction'], inplace=True)


# In[41]:


Fraud_D.head()


# In[42]:


# Model Selection, Training and Validation
# Select Target

y = Fraud_D.fraud_transaction


# In[43]:


X = Fraud_D.drop(['fraud_transaction'], axis = 1)   #Selecting Features


# In[44]:


X


# In[45]:


# Import Ml Algorithms and Implement Them

#import the libraries we will need
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[46]:


## Train test split( training on 80% while testing is 20%)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[47]:


# Initialize each models
LR = LogisticRegression(random_state=42)
KN = KNeighborsClassifier()
DC = DecisionTreeClassifier(random_state=42)
RF = RandomForestClassifier(random_state=42)


# In[48]:


#create list of your model names
models = [LR,KN,DC,RF]


# In[49]:


def plot_confusion_matrix(y_test,prediction):
    cm_ = confusion_matrix(y_test,prediction)
    plt.figure(figsize = (6,4))
    sns.heatmap(cm_, cmap ='coolwarm', linecolor = 'white', linewidths = 1, annot = True, fmt = 'd')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# In[50]:


from sklearn.metrics import confusion_matrix


# In[51]:


#create function to train a model and evaluate accuracy
def trainer(model,X_train,y_train,X_test,y_test):
    #fit your model
    model.fit(X_train,y_train)
    #predict on the fitted model
    prediction = model.predict(X_test)
    #print evaluation metric
    print('\nFor {}, Accuracy score is {} \n'.format(model.__class__.__name__,accuracy_score(prediction,y_test)))
    print(classification_report(y_test, prediction)) #use this later
    plot_confusion_matrix(y_test,prediction)


# In[52]:


#loop through each model, training in the process
for model in models:
    trainer(model,X_train,y_train,X_test,y_test)

# Interpretation of the result
# The Decision Tree model with default parameters yields 99.96% accuracy on training data.
# Precision Score: This means that 82% of all the things we predicted came true. that is 82% of clients transactions was detected to be a fraudulent transaction.

# Recall Score: In all the actual positives, we only predicted 82% of it to be true.

# Random Forest Tree model with default parameters yields 99.97% accuracy on training data.
# Precision Score: This means that 99% of all the things we predicted came true. that is 99% of clients transactions was detected to be a fraudulent transaction.

# Recall Score: In all the actual positives, we only predicted 81% of it to be true.

# Both the Decision Tree and Random Forest models outperform the Logistic Regression and K-Nearest Neighbors model by a wide margin. Since they both have similar recall scores, we should perform a cross-validation of the two models so we may declare which is the best performer with more certainty.
# In[ ]:


# # Cross Validation

# # Importing the library to perform cross-validation
# from sklearn.model_selection import cross_validate

# # Running the cross-validation on both Decision Tree and Random Forest models; specifying recall as the scoring metric
# DC_scores = cross_validate(DC, X_test, y_test, scoring='recall_macro')
# RF_scores = cross_validate(RF, X_test, y_test, scoring='recall_macro')

# # Printing the means of the cross-validations for both models
# print('Decision Tree Recall Cross-Validation:', np.mean(DC_scores['test_score']))
# print('Random Forest Recall Cross-Validation:', np.mean(RF_scores['test_score']))


# In[ ]:


# Conclusion
# Upon training and evaluating our classification model, we found that the Random Forest model performed the best by a narrow margin.

# Therefore, Random Forest performs best with recall cross-validation accuracy of 87% which is important for our problem statement where false negative is our priority

# Recommendation
# Transaction History and Frequency - if unaccounted transactions occurs frequently we should confirm genuinity of the transaction with the customer

# Repeated wrong PIN or Password - We should halt the transaction and alert the customer immediately.

# Make customers to change PIN or password often

# Instruct user to use own mobile or computers while doing transactions to avoid phishing attacks

# Increased cybersecurity for banking websites and mobile applications

# Two factor authentication for transaction

# Ensure that blossom bank hire a data engineer that will ensure the dataset is accurate, balanced for proper EDA as there are too many outliers in this data set. This will enable the business to build machime learning models that predict outcomes more accurately with better performance.


# In[59]:


# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Load Dataset
file_path = "Online Payment Fraud Detection.csv"  # Update with correct path
Fraud_D = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Remove the last column
Fraud_D = Fraud_D.iloc[:, :-1]

# Rename Columns
Fraud_D.columns= ["step", "type", "amount", "customer_starting_transaction", "bal_before_transaction", 
                  "bal_after_transaction", "recipient_of_transaction", "bal_of_recepient_before_transaction", 
                  "bal_of_receipient_after_transaction", "fraud_transaction"]

# One-Hot Encoding for 'type' Column
dummies = pd.get_dummies(Fraud_D['type'], prefix='type')
Fraud_D = pd.concat([Fraud_D, dummies], axis=1)
Fraud_D.drop(['type', 'customer_starting_transaction', 'recipient_of_transaction'], axis=1, inplace=True)

# Reduce the dataset size (e.g., 10% sample)
Fraud_D = Fraud_D.sample(frac=0.1, random_state=42)

# Define Features and Target
X = Fraud_D.drop(['fraud_transaction'], axis=1)
y = Fraud_D['fraud_transaction']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model - Example with Naive Bayes for fast execution
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print Accuracy and Classification Report
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix in your desired format
cm_ = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_, cmap='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




