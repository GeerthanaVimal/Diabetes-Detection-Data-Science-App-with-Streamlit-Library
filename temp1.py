#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[48]:


#pip install streamlit


# In[49]:


from PIL import Image
import streamlit as st


# In[50]:


st.header("Diabetes Detection App Using ML") #Building The App Environment


# In[51]:


img=Image.open('E:\Data Science Masterclass Program\diab.jpeg')
st.image(img)


# In[52]:


#img


# In[53]:


data=pd.read_csv("E:\Data Science Masterclass Program\diabetes.csv")


# In[54]:


st.subheader("Trained Data Information")


# In[55]:


st.dataframe(data)


# In[56]:


st.subheader("Summary of Trained Data")


# In[57]:


st.write(data.describe())


# In[58]:


x=data.iloc[:,:8].values       #Machine Learning
y=data.iloc[:,8].values


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[60]:


model=RandomForestClassifier(n_estimators=500)


# In[61]:


model.fit(x_train,y_train)


# In[62]:


y_pred=model.predict(x_test)


# In[63]:


st.subheader("Accuracy of The Trained Model")


# In[64]:


st.write(accuracy_score(y_test,y_pred))


# In[65]:


def user_inputs():                             #User Inputs
 preg=st.slider("Pregnancies",0,20,0)
 glu=st.slider("Glucose",0,200,0)
 bp=st.slider("Blood Pressure",0,130,0)
 sthick=st.slider("Skin Thickness",0,100,0)
 ins=st.slider("Insulin",0.0,1000.0,0.0)
 bmi=st.slider("BMI",0.0,70.0,0.0)
 dpf=st.slider("DPF",0.000,3.000,0.000)
 age=st.slider("Age",0,100,0)
 
 input_dict={
 "Pregnancies":preg,
 "Glucose":glu,
 "Blood Pressure":bp,
 "Skin Thickness":sthick,
 "Insulin":ins,
 "BMI":bmi,
 "DPF":dpf,
 "Age":age
 }
 
 return pd.DataFrame(input_dict,index=[0])
ui=user_inputs()


# In[66]:


st.subheader("User Inputs")


# In[67]:


st.write(ui)


# In[68]:


st.subheader("Predictions (0 - Non Diabetes, 1 - Diabetes)")      #Predictions for User Inputs


# In[69]:


st.write(model.predict(ui))


# In[ ]:




