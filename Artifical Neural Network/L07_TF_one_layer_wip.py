#!/usr/bin/env python
# coding: utf-8

# # Deep Neural Networks 
# ## Lecture 03
# 
# ## Implementation of Perceptron
# 

# ## 1. Import Statements

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
 
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# gpus = tf.config.list_physical_devices('GPU')
# try: 
#     for g in gpus:
#         tf.config.experimental.set_memory_growth(g,True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus),'Physical GPUs',len(logical_gpus),'Logical GPUs')
# except:
#     print("Invalid Device")


# ## 2. Setup Global Parameters

# In[3]:


###----------------
### Some parameters
###----------------

# Directory locations
inpDir = '../../input'
outDir = '../output'

RANDOM_STATE = 24 # REMEMBER: to remove at the time of promotion to production
np.random.seed(RANDOM_STATE)
rng = np.random.default_rng(seed = RANDOM_STATE) # Set Random Seed for reproducible  results

NOISE = 0.2
EPOCHS = 201 # number of epochs
ALPHA = 0.1  # learning rate
N_SAMPLES = 1000
TEST_SIZE = 0.2

# parameters for Matplotlib
params = {'legend.fontsize': 'medium',
          'figure.figsize': (15, 6),
          'axes.labelsize': 'large',
          'axes.titlesize':'large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'
         }

plt.rcParams.update(params)

CMAP = plt.cm.coolwarm
plt.style.use('seaborn-v0_8-darkgrid') # plt.style.use('ggplot')


# ## 3. Generate Data Set
# <div style="font-family: Arial; font-size:1.2em;color:white;">
# Sklearn's dataset generator is good source of data for learning. To keep the example simple, I'll suggest  <a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html">make_moon</a> dataset generator.
# </div>

# In[4]:


X,y = datasets.make_moons(n_samples=N_SAMPLES,shuffle=True,noise=NOISE,random_state=RANDOM_STATE)
X.shape,y.shape


# ## 4. Visualization
# <p style="font-family: Arial; font-size:1.2em;color:white;">
# DataFrames are easier to visualize
# </p>

# In[5]:


data_df=pd.DataFrame(X,columns=['A','B'])
data_df['target']=y
data_df.head()


# In[6]:


data_df.info()


# In[7]:


data_df.describe().T


# In[8]:


data_df['target'].unique()


# ### 4.1 Different ways of plotting data

# In[9]:


data_df.plot.scatter('A','B',s=10,c='target',cmap=CMAP)


# In[10]:


data_df['target'].value_counts().plot(kind='bar')


# In[11]:


sns.scatterplot(x='A',y='B',data=data_df,hue='target');


# <div style="font-family: Arial; font-size:1.2em;">
#     We will keep 10%, i.e. 100 records for testing and remaining records will be used in training. Note that the data is already random.
# </div>

# #### test Train Split

# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)

X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Model

# In[13]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation='tanh'),     # Hidden Layer with four nodes
    tf.keras.layers.Dense(2)                        # Output Layer with two nodes
])


# #### Loss function

# In[14]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[15]:


model.compile(optimizer='rmsprop',
              loss=loss_fn,
              metrics=['accuracy'])


# In[16]:


history = model.fit(X_train,y_train,
                     validation_data=[X_test,y_test],
                     epochs=EPOCHS,
                     verbose=2)


# In[17]:


res_df = pd.DataFrame(history.history)
res_df


# In[18]:


res_df.plot(y=['loss','val_loss'])


# In[19]:


res_df.plot(y=['accuracy','val_accuracy'])


# ## Prediction

# In[22]:


model.evaluate(X_train,y_train)


# In[23]:


model.evaluate(X_test,y_test)


# #### Prediction: Train 

# In[24]:


y_pred = model.predict(X_train)
y_pred


# In[26]:


accuracy_score(np.argmax(y_pred,axis=1),y_train)


# #### Prediction: Test

# In[28]:


y_pred = model.predict(X_test)
y_pred


# In[29]:


accuracy_score(np.argmax(y_pred,axis=1),y_test)


# In[39]:


pred_func = model.predict(X_train)


# ## Decision Boundary

# In[30]:


def fn_plot_decision_boundary(pred_func,X_tr,y_tr,X_ts,y_ts):
    '''
        Attrib:
           pred_func : function based on predict method of the classifier
           X_tr : train feature matrix
           y_tr : train labels
           X_ts : test feature matrix
           y_ts : test labels
       Return:
           None
    '''
    
    # Set min and max values and give it some padding
    xMin, xMax = X_tr[:, 0].min() - .05, X_tr[:, 0].max() + .05
    yMin, yMax = X_tr[:, 1].min() - .05, X_tr[:, 1].max() + .05
    
    # grid size for mesh grid
    h = 0.01
    
    # Generate a grid of points with distance 'h' between them
    xx, yy = np.meshgrid(np.arange(xMin, xMax, h), np.arange(yMin, yMax, h))
    
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    
    # Make its shape same as that of xx 
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_axes(111)
    
    # Now we have Z value corresponding to each of the combination of xx and yy
    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=CMAP) #, alpha = 0.8
    
    # Plotting scatter for train data
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=np.argmax(y_tr,axis=1),
                                  s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    
    
    # Plotting scatter for test data
    ax.scatter(X_ts[:, 0], X_ts[:, 1], c=np.argmax(y_ts,axis=1),
                                  s=150, marker = '*',edgecolor='k', cmap=plt.cm.inferno )

    
    


# In[31]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[40]:


# loss_df = pd.DataFrame(hist)

fn_plot_decision_boundary(lambda x: pred_func, X_train, y_train, X_test, y_test) # plot decision boundary for this plot

plt.title("Decision Boundary");


# ## Tracking
# <div style="font-family: Arial; font-size:1.2em;color:black;">
# Lets track the results across various implementations...
# 
#  |#|Implementation|Training Accuracy|Testing Accuracy|Remarks|
#  |:-:|---|---|---|---|
#  |1|Simple Perceptron|0.83111|0.89000||

# ## Notes:
# <img src="images/dnn_nb_s03_fig1.png" width='350' align = 'left'>
# <img src="images/dnn_nb_s03_fig2.png" width='350' align = 'right'>

# ## A note on Loss Function
# <div style="font-family: Arial; font-size:1.2em;">
#     <p>In logistic regression we are looking for if it is correct class or not. </p> 
#     <p>For example, we want to know if there is a car in the picture or not. So the output is probability of a car in the picture.</p>
#     <p><b>Mathematically speaking:</b></p>
#     <p>$\hat{y} = p(y=1|x)$ i.e. given training sample $x$, we want to know probability of $y$ being 1.</p>
#     <br>
#     <p><b>Alternatively:</b></p>
#     <p>If there is a car in the picture.  $\Rightarrow$  $y$ = 1 then $p(y|x)$ = $\hat{y}$.</p>
#     <p>If there is <b>no</b> car in the picture.$\Rightarrow$ $y$ = 0 then $p(y|x)$ = 1 - $\hat{y}$.</p>
#     <br>
#     <p>We can summarize two equations as: $p(y|x)$ = $\hat{y}^{y} * (1 - \hat{y}) ^{(1-y)}$</p>
#     <p>Above equation is $\hat{y}$ for y = 1 and (1 - $\hat{y}$) for y = 0.</p>
#     <p>Taking log of above equation:</p>
# 
# $
# \begin{aligned}
# log [ p(y|x) ] & = log[\hat{y}^{y} * (1 - \hat{y}) ^{(1-y)}]\\
# & = y * log(\hat{y}) + (1-y) * log(1 - \hat{y})\\
# \end{aligned}
# $
# <p>Since we aim to minimize above function, add negative sign and our loss function becomes</p>
# 
# $
# \begin{aligned}
# L(\hat{y},y) =  -[y * log\hat{y} + (1-y) * log(1-\hat{y})]\\
# \text{or}\\
# L(a,y) =  - [ y * log ( a ) + ( 1 - y ) * log( 1 - a ) ]\\
# \end{aligned}
# $
# 
# |Case| y |Loss| a |-log(a)|-log(1-a)|
# |:-: |:-:|:-: |:-:|  :-: |   :-:  |
# | 1  | 0 | -log( 1 - a )| 0.000001 |13.8155|**1 e-6**|
# | 2  | 0 | -log( 1 - a )| 0.999999 |1 e-6|**13.8155**|
# | 3  | 1 | -log( a )| 0.000001 |**13.8155**|1 e-6|
# | 4  | 1 | -log( a )| 0.999999 |**1 e-6**|13.8155|
# 
# </div>

# <div style="font-family: Arial; font-size:1.2em;">
#     <p>For binary classification the error = - $y * log(a)$</p>
#     <p>We want to sum it up for all samples in the dataset. Hence:</p>
# 
# $
# \begin{aligned}
# p(\text{all ys | all rows of x}) & =  \Pi_{i=0}^m p(y|x)\\
# log [ p(\text{all ys | all rows of x})] & =  log [ \Pi_{i=0}^m p(y|x) ]\\
# & =  \sum_{i=0}^m log [ p(y|x) ] \\
# & =  \sum_{i=0}^m [ y * log(\hat{y}) + (1-y) * log(1 - \hat{y}) ]\\
# \text{Divide it by m to better scale the costs}\\
# & = \frac{1}{m} * \sum_{i=0}^m [ y * log(\hat{y}) + (1-y) * log(1 - \hat{y}) ]\\
# \end{aligned}
# $

# ### Introducing $\mathrm{sigmoid}$ function for our binary output.
# $$
# \begin{aligned}
# z & = x_1 . w_1 + x_2 . w_2 + b_1 \\
# a & = \hat{y} = \sigma(z)\\
# dz & = (a - y) \\
# db & = dz\\
# b & = b - \alpha . db\\
# dw_1 & = x_1. dz\\
# dw_2 & = x_2.dz\\
# w_1 & = w_1 - \alpha . dw_1\\
# w_2 & = w_1 - \alpha . dw_2\\
# \end{aligned}
# $$
# ### Sigmoid function
# $$
# \begin{align}
# a &= \sigma(z)\\
# &= \dfrac{1}{1 + e^{-z}}\\
# \end{align}
# $$
#     <h3>Derivative of sigmoid function</h3>
# $$
# \begin{align}
# \partial{a} &= \partial{(\sigma(z))}\\
# &= \dfrac{\partial}{\partial{z}} \left[ \dfrac{1}{1 + e^{-z}} \right] \\
# &= \dfrac{\partial}{\partial{z}} \left( 1 + \mathrm{e}^{-z} \right)^{-1} \\
# &= -(1 + e^{-z})^{-2}(-e^{-z}) \\
# &= \dfrac{e^{-z}}{\left(1 + e^{-z}\right)^2} \\
# &= \dfrac{1}{1 + e^{-z}\ } \circ \dfrac{e^{-z}}{1 + e^{-z}}  \\
# &= \dfrac{1}{1 + e^{-z}\ } \circ \dfrac{(1 + e^{-z}) - 1}{1 + e^{-z}}  \\
# &= \dfrac{1}{1 + e^{-z}\ } \circ \left[ \dfrac{1 + e^{-z}}{1 + e^{-z}} - \dfrac{1}{1 + e^{-z}} \right] \\
# &= \dfrac{1}{1 + e^{-z}\ } \circ \left[ 1 - \dfrac{1}{1 + e^{-z}} \right] \\
# &= \sigma(z) \circ (1 - \sigma(z))\\
# &= a \circ (1 - a)
# \end{align}
# $$
#     </div>

# In[ ]:




