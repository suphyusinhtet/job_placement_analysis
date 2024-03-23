import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import requests
import pickle
from sklearn.model_selection import train_test_split
rfmodel= st.sidebar.checkbox('Random Forest')
url = 'https://raw.githubusercontent.com/suphyusinhtet/job_placement_analysis/main/job_placement.csv'
data = pd.read_csv(url)
if rfmodel:
    
    with st.form("my_form1"):
        
        st.title('Prediction of Salary')
        
        st.subheader("Please Choose Person Circustances")
        gendergp = st.selectbox("What's your Gender:", ('Female', 'Male'))
        agegp = st.radio("What's your Age:", ['23', '24', '25','26'])

        # streamgp = st.selectbox("What's your major:", ('Computer Science', 'Electrical Engineering', 'Mechanical Engineering','Information Technology','Electronics and Communication'))
        gpagp = st.selectbox("What's your GPA", ('3.4','3.5','3.6','3.7','3.8','3.9'))
        
        expgp = st.radio("What is your experience:", ['1','2','3'])

        """You selected"""
        st.write("Gender:", gendergp + ", Age: " + agegp)
        # st.write("Stream:",streamgp )
        st.write("GPA: ",  gpagp + ",  Work Experience: " + expgp)
    
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            inputdata = {'gender': gendergp,
                        'age': agegp, 
                         'gpa': gpagp,
                        'years_of_experience': expgp}
            features = pd.DataFrame(inputdata, index=[0])
            st.write(features)
            
            # features_dummy = pd.get_dummies(features)
            # st.write(features_dummy)
            
            # le = LabelEncoder()
            # for i in ["gender", "stream"]:
            #     features[i] = le.fit_transform(features[i])
            gender_map = {'Male':1, 'Female':0}
            features['gender'] = features['gender'].map(gender_map)

            # stream_map = {'Computer Science':0, 'Electrical Engineering':1, 'Electronics and Communication':2, 'Information Technology':3, 'Mechanical Engineering':4 }
            # features['stream'] = features['stream'].map(stream_map)
            st.write(features)
            placement_map = {'Placed':1, 'Not Placed': 0}
            data['placement_status'] = data['placement_status'].map(placement_map)
            # data['stream'] = data['stream'].map(stream_map)
            data['gender'] = data['gender'].map(gender_map)
        #     ##################################################
            scaler = MinMaxScaler()
            st.write(data)
            features_2 = ["gender", "age", "gpa", "years_of_experience"]
            # data_1 = data.loc[data["placement_status"] == 1].values
            data_2 = data.loc[data["placement_status"] == 1, features_2+["salary"]].values
            st.write(data_2)
            data = scaler.fit_transform(data_2)
            # st.write(data)
            X = data[:, :-1]
            Y = data[:, -1]
            # load model
            url = 'https://raw.githubusercontent.com/suphyusinhtet/job_placement_analysis/main/randon_forest_model.sav'
            # loaded_model = pickle.load(open(filename, "rb"))
            # Download the pickle file
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Load the pickle file
                loaded_model = pickle.loads(response.content)
            else:
                # Handle the case when the request fails
                print("Failed to download the pickle file")
            
            
            predicted_salary = loaded_model.predict(features)
            combined_data = np.concatenate((features, predicted_salary.reshape(-1, 1)), axis=1)
            st.write(combined_data)
            a = scaler.inverse_transform(combined_data)
            st.write(a)
            # Extract the prediction from the original_combined_data
            original_pred_result = a[:, -1]
            
            # Print the original prediction
            print("Original prediction:", original_pred_result)
            
            st.write("Salary:", original_pred_result)
        
