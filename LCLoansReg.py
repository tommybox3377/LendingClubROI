#!/usr/bin/env python
# coding: utf-8

# data source : https://www.kaggle.com/wendykan/lending-club-loan-data

# In[2]:


# Import required packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np

from statistics import mean
import re


# In[3]:


# Define custom method
def desc_encode(desc):
    index = desc[0]
    desc = desc[1]
    l = 0
    entries = 0
    if type(desc["desc"]) == str: 
        matches = re.findall(r"(Borrower added on .* >)(.*)", desc["desc"].replace("<br>", "\n"))
        entries = (len(matches))
        for match in matches:
            l += len(match[1])
    return (index, l, entries)


# In[4]:


# turn csv data file to pandas DataFrame
full_raw_data = pd.read_csv(r"/home/tommybox/Desktop/lending-club-loan-data/loan.csv", low_memory=False)


# In[5]:


# pull required columns out of full_raw_data DataFrame
cols = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "installment",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "loan_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "int_rate",   
    "total_pymnt",
    "desc"    
]
raw_data = full_raw_data[cols]


# In[6]:


# Clean Data
data = raw_data[:]
# return int of months of the loan duration
data["term"] = data["term"].apply(lambda x: int(re.findall("\d+", x)[0]))
# Remove unwanted loans based on homeownership, , loan status, loan duration, and funding completion
removed_unwanted_rows = data[
    ((data["home_ownership"] == "MORTGAGE") | (data["home_ownership"] == "RENT") | (data["home_ownership"] == "OWN")) &
    ((data["loan_status"] == "Charged Off") | (data["loan_status"] == "Fully Paid") | (data["loan_status"] == "Default")) &
    (data["term"] == 36) &
    (data["loan_amnt"] == data["funded_amnt"])
    ]
removed_unwanted_cols = removed_unwanted_rows.drop(["term", "funded_amnt"], axis=1)
# Clean Employment length to return int 0-10
removed_unwanted_cols["emp_length"].fillna("0", inplace=True)
removed_unwanted_cols["emp_length"] = pd.to_numeric(removed_unwanted_cols["emp_length"].apply(lambda x: re.findall("\d+", x)[0]))
# Create required datapoints 
removed_unwanted_cols["payment_ratio"] = removed_unwanted_cols["annual_inc"]/removed_unwanted_cols["installment"]
removed_unwanted_cols["roi"] = removed_unwanted_cols["total_pymnt"]/removed_unwanted_cols["loan_amnt"]
# One Hot Encode Home Ownership ans purpose
own = pd.get_dummies(removed_unwanted_cols["home_ownership"])
purpose = pd.get_dummies(removed_unwanted_cols["purpose"])
# Quantify description
desc_code = pd.DataFrame(list(map(desc_encode, removed_unwanted_cols.iterrows())), columns=["indx", "desc_len", "desc_entry_count"]).set_index('indx')
# Final concat/clean
raw_final = removed_unwanted_cols.drop(["installment", "purpose", "desc", "home_ownership", "loan_status", "total_pymnt"], axis=1)
final = raw_final.join(own).join(purpose).join(desc_code)
final = final.dropna()


# In[7]:


goal = final["roi"].mean()
print(goal)


# In[8]:


# Manual Train/Set Split
shuffled_data = final.sample(frac=1)
split = .75
split_int = int(len(shuffled_data)*split)
test = shuffled_data[split_int:]
train = shuffled_data[:split_int]


# In[9]:


# Remove outliers
drop = train[train["annual_inc"]>200000].index
train = train.drop(drop)

drop = train[train["payment_ratio"]>200000].index
train = train.drop(drop)

drop = train[train["desc_len"]>2000].index
train = train.drop(drop)


# In[10]:


X = train.drop(["roi"], axis=1)
y = train["roi"]


# In[11]:


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[12]:


model = SGDRegressor()
model.fit(X, y)


# In[13]:


# Scale/Prep test Data
test_input = test.drop(["roi"], axis=1)
X_test = scaler.transform(test_input)
y_test = test["roi"]


# In[14]:


y_pred = model.predict(X_test)


# In[15]:


# Create a DataFrame to find a threshold that returns best ROI on a resonable percent of loans
log = []
ln = len(y_pred)
for thr in np.linspace(.9, 1.2, num=100):
    l = []
    for i, pred in enumerate(y_pred):
        if pred > thr:
            l.append(test.iloc[i]["roi"])

    log.append((thr, mean(l) if len(l) != 0 else 0, len(l), len(l)/ln))    

logs = pd.DataFrame(log, columns = ["threshold", "roi_avg", "amount", "amt_percent"])


# In[17]:


logs.plot.line(y="roi_avg", x="threshold")


# In[18]:


logs.plot.line(x="threshold", y="amt_percent")


# In[19]:


thr = 1.077
approved_loans= test[y_pred>thr]


# In[20]:


approved_loans.describe()


# In[ ]:




