from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Define lists of options
countries = ['Turkey', 'Australia', 'Spain', 'Indonesia', 'United Kingdom', 'Italy', 'Pakistan', 'Austria', 'Netherlands', 'India', 'France', 'Brazil', 'China', 'Singapore', 'Argentina', 'Poland', 'Germany', 'Canada', 'Thailand', 'Hungary', 'Malaysia', 'Denmark', 'Taiwan', 'Russia', 'Nigeria', 'Belgium', 'South Korea', 'Chile', 'Ireland', 'Philippines', 'Greece', 'Mexico', 'Montenegro', 'United States', 'Bangladesh', 'Japan', 'Slovenia', 'Czechia', 'Sweden', 'United Arab Emirates', 'Switzerland', 'Portugal', 'Peru', 'Hong Kong', 'Vietnam', 'Sri Lanka', 'Serbia', 'Norway', 'Romania', 'Kenya', 'Ukraine', 'Israel', 'Slovakia', 'Lithuania', 'Puerto Rico', 'Bosnia & Herzegovina', 'Croatia', 'South Africa', 'Paraguay', 'Others', 'Colombia', 'Uruguay', 'Algeria', 'Finland', 'Guatemala', 'Egypt', 'Malta', 'Bulgaria', 'New Zealand', 'Kuwait', 'Uzbekistan', 'Saudi Arabia', 'Cyprus', 'Estonia', 'Côte d’Ivoire', 'Morocco', 'Tunisia', 'Venezuela', 'Dominican Republic', 'Senegal', 'Costa Rica', 'Kazakhstan', 'Macedonia (FYROM)', 'Oman', 'Laos', 'Ethiopia', 'Panama', 'Belarus', 'Myanmar (Burma)', 'Moldova', 'Bahrain', 'Mongolia', 'Ghana', 'Albania', 'Kosovo', 'Georgia', 'Tanzania', 'Bolivia', 'Cambodia', 'Iraq', 'Jordan', 'Lebanon', 'Ecuador', 'Jamaica', 'Trinidad & Tobago', 'Libya', 'El Salvador', 'Azerbaijan', 'Nicaragua', 'Palestine', 'Réunion', 'Iceland', 'Armenia', 'Uganda', 'Qatar', 'Cameroon', 'Latvia', 'Congo - Kinshasa', 'Kyrgyzstan', 'Honduras', 'Nepal', 'Luxembourg', 'Sudan', 'Yemen', 'Macau']
browsers = ['Chrome', 'Safari', 'Firefox', 'Internet Explorer', 'Edge', 'Android Webview', 'Safari (in-app)', 'Opera Mini', 'Opera', 'UC Browser', 'YaBrowser', 'Coc Coc', 'Amazon Silk', 'Android Browser', 'Mozilla Compatible Agent', 'MRCHROME', 'Maxthon', 'BlackBerry', 'Nintendo Browser']
subcontinents = ['Western Asia', 'Australasia', 'Southern Europe', 'Southeast Asia', 'Northern Europe', 'Southern Asia', 'Western Europe', 'South America', 'Eastern Asia', 'Eastern Europe', 'Northern America', 'Western Africa', 'Central America', 'Eastern Africa', '(not set)', 'Caribbean', 'Southern Africa', 'Northern Africa', 'Central Asia', 'Middle Africa', 'Melanesia', 'Micronesian Region', 'Polynesia']
operating_systems = ['Windows', 'Macintosh', 'Linux', 'Android', 'iOS', 'Chrome OS', 'BlackBerry', '(not set)', 'Samsung', 'Windows Phone', 'Xbox', 'Nintendo Wii', 'Firefox OS', 'Nintendo WiiU', 'FreeBSD', 'Nokia', 'NTT DoCoMo', 'Nintendo 3DS', 'SunOS', 'OpenBSD']
mediums = ['organic', 'referral', 'cpc', 'affiliate', 'cpm']
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

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract data from form
        hits = int(request.form.get('hits', 0))
        pageviews = int(request.form.get('pageviews', 0))
        visitNumber = int(request.form.get('visitNumber', 0))
        country = request.form.get('country')
        continent = request.form.get('continent')
        browser = request.form.get('browser')
        subContinent = request.form.get('subContinent')
        operatingSystem = request.form.get('operatingSystem')
        medium = request.form.get('medium')
        
        # Prepare input data
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
        
        try:
            # Predict using the model
            log_prediction = model.predict(input_data)
            predicted_revenue = np.exp(log_prediction[0])  # Using np.exp to revert the log transformation
            predicted_revenue = 1000000*(max(0, predicted_revenue))  # Convert to millions and ensure no negative values
            
            return render_template('index.html', predicted_revenue=f"{predicted_revenue:.2f} ",
                                  countries=countries, continents=continents, browsers=browsers,
                                  subcontinents=subcontinents, operating_systems=operating_systems,
                                  mediums=mediums)
        except Exception as e:
            return render_template('index.html', error=str(e),
                                  countries=countries, continents=continents, browsers=browsers,
                                  subcontinents=subcontinents, operating_systems=operating_systems,
                                  mediums=mediums)
    
    return render_template('index.html',
                           countries=countries, continents=continents, browsers=browsers,
                           subcontinents=subcontinents, operating_systems=operating_systems,
                           mediums=mediums)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
