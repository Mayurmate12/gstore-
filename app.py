import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mysql.connector
from mysql.connector import Error

# Define lists of options
countries = ['Turkey', 'Australia', 'Spain', 'Indonesia', 'United Kingdom',
       'Italy', 'Pakistan', 'Austria', 'Netherlands', 'India', 'France',
       'Brazil', 'China', 'Singapore', 'Argentina', 'Poland', 'Germany',
       'Canada', 'Thailand', 'Hungary', 'Malaysia', 'Denmark', 'Taiwan',
       'Russia', 'Nigeria', 'Belgium', 'South Korea', 'Chile', 'Ireland',
       'Philippines', 'Greece', 'Mexico', 'Montenegro', 'United States',
       'Bangladesh', 'Japan', 'Slovenia', 'Czechia', 'Sweden',
       'United Arab Emirates', 'Switzerland', 'Portugal', 'Peru',
       'Hong Kong', 'Vietnam', 'Sri Lanka', 'Serbia', 'Norway', 'Romania',
       'Kenya', 'Ukraine', 'Israel', 'Slovakia', 'Lithuania',
       'Puerto Rico', 'Bosnia & Herzegovina', 'Croatia', 'South Africa',
       'Paraguay', 'Others', 'Colombia', 'Uruguay', 'Algeria', 'Finland',
       'Guatemala', 'Egypt', 'Malta', 'Bulgaria', 'New Zealand', 'Kuwait',
       'Uzbekistan', 'Saudi Arabia', 'Cyprus', 'Estonia', 'Côte d’Ivoire',
       'Morocco', 'Tunisia', 'Venezuela', 'Dominican Republic', 'Senegal',
       'Costa Rica', 'Kazakhstan', 'Macedonia (FYROM)', 'Oman', 'Laos',
       'Ethiopia', 'Panama', 'Belarus', 'Myanmar (Burma)', 'Moldova',
       'Bahrain', 'Mongolia', 'Ghana', 'Albania', 'Kosovo', 'Georgia',
       'Tanzania', 'Bolivia', 'Cambodia', 'Iraq', 'Jordan', 'Lebanon',
       'Ecuador', 'Jamaica', 'Trinidad & Tobago', 'Libya', 'El Salvador',
       'Azerbaijan', 'Nicaragua', 'Palestine', 'Réunion', 'Iceland',
       'Armenia', 'Uganda', 'Qatar', 'Cameroon', 'Latvia',
       'Congo - Kinshasa', 'Kyrgyzstan', 'Honduras', 'Nepal',
       'Luxembourg', 'Sudan', 'Yemen', 'Macau']

browsers = [
    'Chrome', 'Safari', 'Firefox', 'Internet Explorer', 'Edge',
    'Android Webview', 'Safari (in-app)', 'Opera Mini', 'Opera',
    'UC Browser', 'YaBrowser', 'Coc Coc', 'Amazon Silk', 'Android Browser',
    'Mozilla Compatible Agent', 'MRCHROME', 'Maxthon', 'BlackBerry',
    'Nintendo Browser'
]

subcontinents = [
    'Western Asia', 'Australasia', 'Southern Europe', 'Southeast Asia',
    'Northern Europe', 'Southern Asia', 'Western Europe',
    'South America', 'Eastern Asia', 'Eastern Europe',
    'Northern America', 'Western Africa', 'Central America',
    'Eastern Africa', '(not set)', 'Caribbean', 'Southern Africa',
    'Northern Africa', 'Central Asia', 'Middle Africa', 'Melanesia',
    'Micronesian Region', 'Polynesia'
]

operating_systems = [
    'Windows', 'Macintosh', 'Linux', 'Android', 'iOS', 'Chrome OS',
    'BlackBerry', '(not set)', 'Samsung', 'Windows Phone', 'Xbox',
    'Nintendo Wii', 'Firefox OS', 'Nintendo WiiU', 'FreeBSD', 'Nokia',
    'NTT DoCoMo', 'Nintendo 3DS', 'SunOS', 'OpenBSD'
]

mediums = [
    'organic', 'referral', 'cpc', 'affiliate', 'cpm'
]

continents = ['Asia', 'Oceania', 'Europe', 'Americas', 'Africa']

# Load the model and encoders
def load_model_and_encoders():
    # Load the RandomForestRegressor model
    with open('rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load the encoders
    with open('label_encoders.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    
    # Load feature names
    with open('feature_names.pkl', 'rb') as feature_file:
        feature_names = pickle.load(feature_file)
    
    return model, encoders, feature_names

model, encoders, feature_names = load_model_and_encoders()

# Connect to MySQL database
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Mayur@123',
            database='project'
        )
        return connection
    except Error as e:
        st.error(f"Error: {str(e)}")
        return None

def save_prediction(hits, pageviews, visitNumber, country, continent, browser, subContinent, operatingSystem, medium, predicted_revenue):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO predictions (hits, pageviews, visitNumber, country, continent, browser, subContinent, operatingSystem, medium, predicted_revenue) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (hits, pageviews, visitNumber, country, continent, browser, subContinent, operatingSystem, medium, predicted_revenue)
            )
            connection.commit()
            cursor.close()
            st.success("Prediction saved to database!")
        except Error as e:
            st.error(f"Database error: {str(e)}")
        finally:
            connection.close()

# Streamlit UI
st.set_page_config(page_title="Revenue Prediction", layout="wide")

# Custom CSS for background image and styling
st.markdown("""
    <style>
    .main {
        background-image: url('background.jpg');
        background-size: cover;
        background-position: center;
        color: white;
    }
    .streamlit-expanderHeader {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Google Store Revenue Prediction")

# User input fields
hits = st.number_input("Hits", value=0, min_value=0)
pageviews = st.number_input("Pageviews", value=0, min_value=0)
visitNumber = st.number_input("Visit Number", value=0, min_value=0)
country = st.selectbox("Country", countries)
continent = st.selectbox("Continent", continents)
browser = st.selectbox("Browser", browsers)
subContinent = st.selectbox("Sub Continent", subcontinents)
operatingSystem = st.selectbox("Operating System", operating_systems)
medium = st.selectbox("Medium", mediums)

# Prepare input for model
def prepare_input():
    input_data = pd.DataFrame({
        'hits': [hits],
        'pageviews': [pageviews],
        'visitNumber': [visitNumber],
        'country': [country],
        'continent': [continent],
        'browser': [browser],
        'subContinent': [subContinent],
        'operatingSystem': [operatingSystem],
        'medium': [medium]
    })
    
    input_data.replace('', np.nan, inplace=True)
    
    for feature in encoders:
        if feature in input_data.columns:
            encoder = encoders[feature]
            if 'Unknown' not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, 'Unknown')
            input_data[feature] = input_data[feature].fillna('Unknown')
            input_data[feature] = input_data[feature].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
            input_data[feature] = encoder.transform(input_data[feature])
    
    # Reorder columns to match feature names used during model training
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    
    return input_data

input_data = prepare_input()

if st.button("Predict"):
    try:
        # Predict using the model
        log_prediction = model.predict(input_data)
        # Convert the log-transformed prediction back to original scale
        predicted_revenue = (1000000*np.exp(log_prediction[0]))  # Using np.exp to revert the log transformation
        predicted_revenue = max(0, predicted_revenue)  # Ensure no negative values
        st.success(f"Predicted Revenue: {predicted_revenue:.2f}")
        # Save the prediction to the database
        save_prediction(hits, pageviews, visitNumber, country, continent, browser, subContinent, operatingSystem, medium, predicted_revenue)
    except Exception as e:
        st.error(f"Error: {str(e)}")
