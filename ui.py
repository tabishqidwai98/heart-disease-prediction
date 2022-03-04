from altair.vegalite.v4.api import value
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from disease_prediction import predict
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from db import Prediction
def open_db():
    engine = create_engine("sqlite:///db.sqlite3")
    Session = sessionmaker(bind=engine)
    return Session()

menu = ['About Project','View Data','View Visualization','Predict Disease','Previous Predictions']
st.title("Heart disease prediction")
st.sidebar.image('sidebar.png', use_column_width=True)
st.sidebar.title("HEART DISEASE PREDICTION")
ch = st.sidebar.selectbox("choose an option",menu)
if ch == 'About Project':
    st.header("About this project")
    st.image('1.jpg')
    st.subheader("What is heart disease?")
    st.markdown("The term “heart disease” refers to several types of heart conditions. The most common type of heart disease in the United States is coronary artery disease (CAD), which affects the blood flow to the heart. Decreased blood flow can cause a heart attack.")
    
    st.subheader("Who gets heart disease?")
    st.markdown("""Heart disease is the leading cause of death in the United States, according to the Centers for Disease Control and Prevention (CDC)Trusted Source. In the United States, 1 in every 4 deaths in is the result of a heart disease. That's about 610,000 people who die from the condition each year.

Heart disease doesn't discriminate. It's the leading cause of death for several populations, including white people, Hispanics, and Black people. Almost half of Americans are at risk for heart disease, and the numbers are rising. Learn more about the increase in heart disease rates.

While heart disease can be deadly, it's also preventable in most people. By adopting healthy lifestyle habits early, you can potentially live longer with a healthier heart.""")
    
    st.subheader("What are the different types of heart disease?")
    st.markdown("""
    Heart disease encompasses a wide range of cardiovascular problems. Several diseases and conditions fall under the umbrella of heart disease. Types of heart disease include:
    """)
    st.markdown("""
        - Arrhythmia. An arrhythmia is a heart rhythm abnormality.
        - Atherosclerosis. Atherosclerosis is a hardening of the arteries.
        - Cardiomyopathy. This condition causes the heart's muscles to harden or grow weak.
        - Congenital heart defects. Congenital heart defects are heart irregularities that are present at birth.
        - Coronary artery disease (CAD). CAD is caused by the buildup of plaque in the heart's arteries. It's sometimes called ischemic heart disease.
        - Heart infections. Heart infections may be caused by bacteria, viruses, or parasites.
    """)

    st.subheader("How is heart disease diagnosed?")
    st.markdown("""
        - Physical exams and blood tests
        - Noninvasive tests
        - Invasive tests
    """)

    st.header("Our Approach")
    st.write("We are trying to predict heart disease based on the medical test and history of the patients. We have obtained an accuracy of 86.8%. There is a analysis section of the project for better understanding of the data. we can viewing previous patients data too. The predict page of this web app is used to predict if the patient will have heart disease in future.")
    st.write(" - There is sidebar on the left side of the web page where in dropdown, u can select different feature options to view and use. On the top right corner, there is a 3 dot button for changing the settings.")

if ch == 'View Data':
    st.image("2.png")
    st.subheader('View dataset of patients')
    df = pd.read_csv('heart.csv')
    attr_ck = st.sidebar.checkbox('view column info')
    st.write(df)
    if attr_ck:
        st.markdown(open('attributes.md').read(),unsafe_allow_html=True)

if ch == 'View Visualization':
    df = pd.read_csv('heart.csv')
    goptions = ['patient distribution count',
                'patient with chest pain',
                'age wise distribution of heart patients',
                'gender wise division of heart patients',
                'patients with heart rate slope',
                'exercise induced angina in patients',
                'thalassemia blood disorder in patients',
                'age vs cholestrol in patients',
                'exercise induced angina vs maximum heart rate achieved in patients',
                'age vs maximum heart rate achieved in patients']
    submenu = st.sidebar.radio('select graph',goptions)
    if submenu == 'patient distribution count':
        fig,ax =plt.subplots()
        sns.countplot(df.target,ax=ax)
        st.pyplot(fig)
    if submenu == 'patinets with chest pain':
        fig,ax =plt.subplots()
        sns.countplot(df.cp,ax=ax)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels([' typical angina','atypical angina','non-anginal pain','asymptomatic'])
        st.pyplot(fig)
    if submenu == 'age wise distribution of heart patients':
        fig,ax= plt.subplots(figsize=(15,8))
        sns.countplot(df.age, hue=df.target,ax=ax)
        st.pyplot(fig)
    if submenu == 'gender wise division of heart patients':
        fig,ax=plt.subplots(figsize=(15,8))
        sns.countplot(df.sex, hue=df.target,ax=ax)
        st.pyplot(fig)
    if submenu =='patients with heart rate slope':
        fig,ax=plt.subplots(figsize=(15,8))
        sns.countplot(df.slope,hue=df.target,ax=ax)
        st.pyplot(fig)
    if submenu =='exercise induced angina in patients':
        fig,ax=plt.subplots(figsize=(15,8))
        sns.countplot(df.exang,hue=df.target,ax=ax)
        st.pyplot(fig)
    if submenu =='thalassemia blood disorder in patients':
        fig,ax=plt.subplots(figsize=(20,8))
        sns.countplot(df.thal,hue=df.target,ax=ax)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['NULL','fixed defect (no blood flow in some part of the heart)','normal blood flow','reversible defect (a blood flow is observed but it is not normal)'])
        st.pyplot(fig)
    if submenu =='age vs cholestrol in patients':
        fig,ax=plt.subplots(figsize=(6,6))
        sns.scatterplot(df.age,df.chol,hue=df.target,ax=ax)
        st.pyplot(fig)
    if submenu =='exercise induced angina vs maximum heart rate achieved in patients':
        fig,ax=plt.subplots(figsize=(5,5))
        sns.scatterplot(df.exang,df.thalach,hue=df.target,ax=ax)
        ax.set_xticks([0,1])
        st.pyplot(fig)
    if submenu =='age vs maximum heart rate achieved in patients':
        fig,ax=plt.subplots(figsize=(10,10))
        sns.scatterplot(df.age,df.thalach,hue=df.target,ax=ax)
        st.pyplot(fig)

if ch =='Predict Disease':

    st.image('4.jpg')
   
    st.sidebar.image('model_report.png',use_column_width=True,caption="our model performance")
    st.subheader("Enter patient details")
    patient_name = st.text_input('Enter patient name')
    age = st.number_input("Age",min_value=1,max_value=100,value=30)
    sex = st.radio("Gender",['Female','Male'])
    cp = st.radio('Area of Chest Pain',['typical angina','atypical angina','non-anginal pain','asymptomatic'])
    trestbps = st.number_input('resting blood pressure (in mm Hg on admission to the hospital)',min_value=80,max_value=250,value=130)
    chol = st.number_input('Cholesterol value',min_value=100,max_value=600,value=230)
    fbs = st.radio('fasting blood sugar > 120 mg/dl',['No','Yes'])
    restecg = st.radio('resting electrocardiographic results',['normal',' T wave inversions and/or ST elevation or depression of > 0.05 mV','showing probable or definite left ventricular hypertrophy by Estes criteria'])
    thalach = st.number_input('maximum heart rate achieved',min_value=70,max_value=210,value=140)
    exang = st.radio('exercise induced angina',['No','Yes'])
    oldpeak = st.number_input('ST depression induced by exercise relative to rest',min_value=0.0,max_value=6.5,value=1.03)
    slope = st.radio('heart rate slope',['upsloping','flat','downsloping'])
    ca = st.radio('number of major vessels (0-4) colored by flourosopy',[0, 2, 1, 3, 4])
    thal = st.radio('thalassemia blood disorder',['no info','no blood flow in some part of the heart', 'normal blood flow', 'a blood flow is observed but it is not normal'])

    
    sex =1 if sex =='Male' else 0
    cp = ['typical angina','atypical angina','non-anginal pain','asymptomatic'].index(cp)
    fbs = 1 if fbs =='Yes' else 0
    restecg = ['normal',' T wave inversions and/or ST elevation or depression of > 0.05 mV','showing probable or definite left ventricular hypertrophy by Estes criteria'].index(restecg)
    exang = 1 if exang =='Yes' else 0
    slope = ['upsloping','flat','downsloping'].index(slope)
    thal = ['no info','no blood flow in some part of the heart', 'normal blood flow', 'a blood flow is observed but it is not normal'].index(thal)
    
    features = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    st.sidebar.subheader('data to be passed in Al Algo')
    st.sidebar.write(features)
    btn = st.button("Make prediction using AI")
    if patient_name and btn:
        
        with st.spinner("please wait while we use algorithm to predict"):
            result = predict(features=features)
            sess = open_db()
            out = 1 if result else 0
            pred = Prediction(patient=patient_name,features=str(features),result=out)
            sess.add(pred)
            sess.commit()
            sess.close()
            if result:
                st.sidebar.success(f"AI prediction : Patient:{patient_name} seem to be healthy")
            else:
                st.sidebar.error(f"AI prediction : Patient:{patient_name} might be suffering from heart disease")
            

if ch =='Previous Predictions':
    sess = open_db()
    preds = sess.query(Prediction).all()
    sess.close()
    if preds:
        p = st.sidebar.radio("Select a patient to view details",preds)
        st.markdown(f'''
        - patient id : {p.id}
        - patient name : {p.patient}
        - patient details : {p.features}
        - patient result : {"No heart disease" if p.result else "heart disease"}
        ''')
    else:
        st.error("No patient data")