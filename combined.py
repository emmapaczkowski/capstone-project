#!/usr/bin/env python
# coding: utf-8

# # Mushroom Classification 

# In[39]:


# import libraries 
import numpy as np 
import pandas as pd
import warnings
import seaborn as sns
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score, cross_validate, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, plot_confusion_matrix, confusion_matrix, roc_curve, roc_auc_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import dataset 
dataset = pd.read_csv('mushrooms.csv')
dataFrame = dataset


# In[3]:


#look at the data head
dataFrame.head()


# In[5]:


#describe dataset
dataFrame.describe()


# In[6]:


#data frame information
dataFrame.info()


# In[7]:


#check for null data
dataFrame.isnull().sum()


# In[8]:


#count rows and columns
dataFrame.shape


# In[9]:


#count edible vs poisonous 
dataFrame['class'].value_counts()


# ## K-Nearest Neighbors Model

# In[11]:


#preprocessing - changing the values to numbers with label encoder.
def Label_enc(feat):
    LabelE = LabelEncoder()
    LabelE.fit(feat)
    print(feat.name,LabelE.classes_)
    return LabelE.transform(feat)


# In[12]:


for col in dataFrame.columns:
    dataFrame[str(col)] = Label_enc(dataFrame[str(col)])


# In[13]:


#data head affter label encoder
dataFrame.head()


# In[14]:


#Split the data to y and x with x is without the class's.
y = dataFrame['class']
X = dataFrame.drop('class', axis=1)


# In[15]:


#data head affter deleting veil-type and droping the class's
X.head()


# In[16]:


#Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# In[17]:


#data head affter StandardScaler
X.head()


# In[18]:


#split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

#data training set
print('Train data: ', len(X_train)/len(X))

#data testing set
print('Test data: ', X_test.shape[0]/y.shape[0])


# In[19]:


#print the quantity of rows and colums in the testing and training sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[20]:


#knn classifier with for loop [1-20] to check the best accuracy of n
for n in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('KNeighborsClassifier: n = {} , Accuracy is: {}'.format(n,knn.score(X_test,y_test)))


# In[21]:


#plot_confusion_matrix of knn 
plot_confusion_matrix(knn, X_test, y_test, display_labels= ['Edible', 'Poisonous'], cmap = "summer", normalize= None)
plt.title('Confusion Matrix KNN')
plt.show()


# In[22]:


#Print Confusion matrix Accuracy of knn
print('Confusion matrix Accuracy is: {}'.format(metrics.accuracy_score(y_test, y_pred)))


# In[23]:


#classification_report of KNN
KNN_REPORT = classification_report(y_test, knn.predict(X_test))
print(KNN_REPORT)


# # Random Forest Classifier Model

# In[25]:


RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)
y_predict = RFC.predict(X_test)
print('RandomForestClassifier Accuracy is: {}'.format(RFC.score(X_test,y_test)))


# In[26]:


#plot_confusion_matrix of RFC 
plot_confusion_matrix(RFC, X_test, y_test, display_labels= ['Edible', 'Poisonous'], cmap = "summer", normalize= None)
plt.title('Confusion Matrix RFC')
plt.show()


# In[27]:


#Print Confusion matrix Accuracy of RFC
print('Confusion matrix Accuracy is: {}'.format(metrics.accuracy_score(y_test, y_predict)))


# In[28]:


#classification_report of RFC
RFC_REPORT = classification_report(y_test, y_predict)
print(RFC_REPORT)


# In[29]:


# provide an encoder for the predictor values
Encoder_X = LabelEncoder() 

# encode each of the columns, fit_transform will return the encoded labels
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
    
# provide an encoder for the response values
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)

# show the encoded predictors
X.head() 


# In[30]:


#get dummy variables
# one-hot encode the categorical data
X = pd.get_dummies(X, columns=X.columns,drop_first=True)

# show the encoded predictors
X.head()


# In[31]:


# split the data using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[32]:


#feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[33]:


#apply PCA from sklearn
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[34]:


# functions to visualize data
def visualization_train(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title("%s Training Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()


# In[35]:


def visualization_test(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title("%s Test Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()


# In[36]:


#functions to evalute model preformance 
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))
        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))


# # Decision Tree Classification Model

# In[37]:


classifier = DT(criterion='entropy',random_state=42)
classifier.fit(X_train,y_train)


# In[38]:


#print score
print_score(classifier,X_train,y_train,X_test,y_test,train=True)


# # Logistic Regrssion

# In[40]:


classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[42]:


print_score(classifier,X_train,y_train,X_test,y_test,train=True)

