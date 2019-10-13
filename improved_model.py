#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# import pandas as pd
# import numpy as np
# %matplotlib inline
# from pandas import Series,DataFrame
# data_train = pd.read_csv("Train.csv")
# data_train

# In[2]:


data_train.info()


# In[3]:


data_train.describe()


# In[4]:


import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"1 is Survived")
plt.ylabel(u"number")


# In[5]:


plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.ylabel(u"number")
plt.title(u"distribution")


# In[6]:


plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age")
plt.grid(b=True,which='major',axis = 'y')
plt.title("Correlation between age and survival")


# In[7]:


plt.subplot2grid((2,3),(1,0),colspan=2)
# kde stands for kernel density estimation
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.ylabel("density")
plt.title("distribution of passenger's age")
plt.legend(("1","2","3"),loc='best')


# In[8]:


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("Embard")
plt.ylabel("number of passenger")
plt.show()


# In[9]:


# Correlation between the passenger attributes and suvival chance
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# Construct data frame by specify column name
df = pd.DataFrame({'survived':Survived_1,'unsurvived':Survived_0})
df.plot(kind='bar',stacked=True) #stack can make the two bar plot stacked
plt.title("survival rate of each class")
plt.xlabel("passenger class")
plt.ylabel("number of passenger")
plt.show()


# In[10]:


# check the impact of gender
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'male':Survived_m,'female':Survived_f})
df.plot(kind='bar',stacked = True)
plt.xlabel('Sex')
plt.ylabel('number of passengers')
plt.title('Gender plot')
plt.show()


# In[11]:


fig = plt.figure(figsize=(10,10))
fig.set(alpha=0.2)
plt.title("Plot survival according to gender and class")


ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',label="female highclass",color = 'blue',stacked=True)
#ax1.set_xticklabels(["survived","unsurvived"],rotation=0)
ax1.legend(['female high class'],loc='best')


ax2 = fig.add_subplot(142,sharey = ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',label="female lowclass",color = 'pink',stacked=True)
#ax2.set_xticklabels(["unsurvived","survived"],rotation=0)
plt.legend(['female low class'],loc='best')


ax3 = fig.add_subplot(143,sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',label="male highclass",color = 'red',stacked=True)
#ax3.set_xticklabels(["unsurvived","survived"],rotation=0)
plt.legend(['male high class'],loc='best')


ax4 = fig.add_subplot(144,sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',label="male lowclass",color = 'orange',stacked=True)
#ax4.set_xticklabels(["unsurvived","survived"],rotation=0)
plt.legend(['male low class'],loc='best')


plt.show()


# In[12]:


fig =plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'Survived':Survived_1,'Unsurvived':Survived_0})
df.plot(kind='bar',stacked = True)
plt.ylabel('Survival density')


# In[13]:


g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)


# In[14]:


data_train.Cabin.value_counts()


# In[15]:


fig = plt.figure()
fig.set(alpha = 0.2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'notnull':Survived_cabin, 'null':Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title('impact of cabin')
plt.xlabel('cabin')
plt.ylabel('number')
plt.show()


# In[16]:


from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    # Save all the numerical values
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    # Divide the passengers' age into known and unknown (as numpy.ndarray)
    
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # y is the passenger ID
    y = known_age[:,0]
    X = known_age[:,1:]
    
    # Fit into RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)
    
    # Use the model to predict the unknow ages
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # Fill the null value with the predicted ages
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()),'Cabin']="No"
    
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


# In[17]:


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix = 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass],axis = 1)
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1, inplace =True)
df


# In[18]:


import sklearn.preprocessing as preprocessing
# Scale the age and fare parameter
scaler = preprocessing.StandardScaler()


# In[19]:


age_scale_param = scaler.fit(df[['Age']])


# In[20]:


df['Age_scaled'] = scaler.fit_transform(df[['Age']],age_scale_param)
fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']],fare_scale_param)
df


# In[21]:


from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')


# In[22]:


# Convert data into numpy format
train_np = train_df.as_matrix()


# In[23]:


# Set up output
y = train_np[:,0]

# Set up input
X = train_np[:,1:]


# In[24]:


# Fit the linear logistic regression model
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)


# In[25]:


clf


# In[26]:


# Process test data
data_test = pd.read_csv('test.csv')
data_test.head()


# In[27]:


data_test.loc[data_test.Fare.isnull(),'Fare']=0


# In[28]:


# Use the same random forest regressor to fill the unknown ages
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()


# In[29]:


X = null_age[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges
data_test.head()


# In[30]:


data_test = set_Cabin_type(data_test)


# In[31]:



dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


# In[32]:


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)
df_test


# In[33]:


# Get the prediction
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')


# In[34]:


predictions = clf.predict(test)


# In[35]:


result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,'Survived':predictions.astype(np.int32)})


# In[36]:


result.to_csv('logistic_regression_predictions.csv')


# In[37]:


result


# In[38]:


# Check the model coefficents and feature
pd.DataFrame({"columns":list(train_df.columns)[1:],"coef":list(clf.coef_.T)})


# In[39]:


from sklearn.model_selection import cross_val_score

clf = linear_model.LogisticRegression(C = 1.0,penalty = 'l1', tol=1e-6)
all_data = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]

print (cross_val_score(clf, X, y, cv=5))


# In[40]:


from sklearn.model_selection import train_test_split
split_train,split_cv = train_test_split(df,test_size=0.3,random_state =42)


# In[41]:


train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# Fit into model
clf = linear_model.LogisticRegression(solver='liblinear',C=1.0,penalty ='l1', tol= 1e-6 )


# In[42]:


clf.fit(train_df.values[:,1:],train_df.values[:,0])


# In[43]:


# predict with cross validation set
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.values[:,1:])
predictions


# In[44]:


origin_data_train = pd.read_csv('train.csv')


# In[45]:


bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]


# In[ ]:





# In[46]:


# Get the learning curve of the baseline model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, x,y,ylim=None, cv=None, n_jobs =1, train_sizes=np.linspace(.05,1.,20),verbose = 0, plot =True):
    # cv:cross-validation 
    
    train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose = verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Sample number")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="training error")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="cv errror")
    
        plt.legend(loc="best")
        
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, "learning curves", X, y)


# In[47]:


from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np=train_df.values

y = train_np[:,0]

X = train_np[:,1:]

# fit into BaggingRegreesor
clf = linear_model.LogisticRegression(C=1.0, penalty='l1',tol = 1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples = 0.8, max_features = 1.0, bootstrap = True, bootstrap_features = False, n_jobs=-1)
bagging_clf.fit(X,y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
results = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
results.to_csv("logistic_regression_bagging_predictions.csv", index=False)


# In[48]:


import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[49]:


train_data = train_df.as_matrix()
test_data = test.as_matrix()


# In[50]:


X = train_data[:,1:]
y = train_data[:,0]


# In[51]:


# Set up 5 models
stack_model = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]


# In[52]:


ntrain = train_data.shape[0] # Sample number of train data
ntest = test_data.shape[0] # sample number of test data


# In[53]:


type(ntrain)


# In[54]:


train_stack = np.zeros((ntrain,5)) # 5 is the number of models
test_stack = np.zeros((ntest,5))
test_stack_i = np.zeros((ntest,5))


# In[55]:


kf = KFold(n_splits = 5)
kf.get_n_splits(X)
KFold(n_splits = 5, random_state=None, shuffle = False)


# In[56]:


for i,clf in enumerate(stack_model):
    for j, (train_index, holdout_index) in enumerate(kf.split(X)):
        X_train, X_holdout = X[train_index], X[holdout_index]
        y_train, y_holdout = y[train_index], y[holdout_index]
        clf.fit(X_train,y_train)
        train_stack[holdout_index,i]=clf.predict(X_holdout)
        test_stack_i[:,j] = clf.predict(test_data)
    test_stack[:,i]=test_stack_i.mean(1)
    


# In[57]:


model = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model.fit(train_stack, y)

y_predict = model.predict(test_stack)


# In[58]:


y_predict


# In[59]:


stacking_results = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':y_predict.astype(np.int32)})


# In[60]:


stacking_results.to_csv("stacking_predictions.csv", index=False)


# In[ ]:




