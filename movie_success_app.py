import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define the feature columns
feature_columns = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'year', 'in_top_50']

# Function to make predictions
def predict_movie_success(features):
    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    features = np.array(features, dtype=float).reshape(1, -1)
    features = scaler.fit_transform(features)  # Note: In a real scenario, use scaler fitted on training data
    
    # Make prediction
    prediction = model.predict(features)
    return 'Successful' if prediction == 1 else 'Not Successful'

# Streamlit app
st.title('Movie Success Prediction')

# Collect user input
budget = st.number_input('Budget (in million $)', min_value=0)
popularity = st.number_input('Popularity', min_value=0.0)
revenue = st.number_input('Revenue (in million $)', min_value=0)
runtime = st.number_input('Runtime (in minutes)', min_value=0)
vote_average = st.number_input('Vote Average (1-10)', min_value=0.0, max_value=10.0, step=0.1)
vote_count = st.number_input('Vote Count', min_value=0)
year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
in_top_50 = st.selectbox('Is the movie in top 50?', options=['Yes', 'No'])

# Convert in_top_50 to binary
in_top_50_binary = 1 if in_top_50 == 'Yes' else 0

# Create a button to make predictions
if st.button('Predict'):
    # Create feature array
    features = [budget, popularity, revenue, runtime, vote_average, vote_count, year, in_top_50_binary]
    
    # Make prediction
    result = predict_movie_success(features)
    
    # Display result
    st.write(f'The movie is predicted to be: {result}')
