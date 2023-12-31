import streamlit as st
import joblib
import datetime

from features import preprocess_user_input, get_starting_airports, get_destination_airports, get_airline_names

# List of model filenames and their corresponding names
model_filenames = [
    r'C:\Users\chant\adv_mla_2023\data_product_with_ml\models\xgb_default.joblib',
    r'C:\Users\chant\adv_mla_2023\data_product_with_ml\models\rf_model.joblib',
    r'C:\Users\chant\adv_mla_2023\data_product_with_ml\models\linear_reg_model_tuned.joblib',
    r'C:\Users\chant\adv_mla_2023\data_product_with_ml\models\elastic_net_model.joblib'
]

# Model names for display
model_names = [
    "XGBoost",
    "Random Forest",
    "Tuned Linear Regression",
    "Elastic Net"
]

# Streamlit UI components for user inputs
st.title("Local Travel Airfare Estimator")

# Dropdown for starting airport
origin_airport = st.selectbox("Origin Airport", get_starting_airports())

# Dropdown for destination airport
destination_airport = st.selectbox("Destination Airport", get_destination_airports())

# Input for flight date
flight_date = st.date_input("Flight Date")

# Checkbox for basic economy
is_basic_economy = st.checkbox("Basic Economy?")

# Checkbox for refundable ticket
is_refundable = st.checkbox("Refundable Ticket?")

# Checkbox for non-stop flight
is_non_stop = st.checkbox("Non-Stop Flight?")

# Dropdown for cabin type
cabin_type = st.selectbox("Cabin Type", ["Coach", "Premium Coach", "Business", "First"])

# Dropdown for airline name
airline = st.selectbox("Airline Name", get_airline_names())

# User validation button
if st.button("Estimate Fare"):
    # The current date
    search_date = datetime.datetime.now().date()

    # Preprocess user inputs
    user_input = preprocess_user_input(
        origin_airport, destination_airport, flight_date, search_date, is_basic_economy,
        is_refundable, is_non_stop, cabin_type, airline
    )

    # Convert user_input to a 2D array
    user_input_2d = [user_input]  # Make it a single instance with multiple features

    # Display loading spinner while predicting fares
    with st.spinner("Predicting fares..."):
        # Initialize a list to store predicted fares
        predicted_fares = []
        
        # Load and use each model to predict fares
        for i, model_filename in enumerate(model_filenames):
            model = joblib.load(model_filename)
            predicted_fare = model.predict(user_input_2d)
            model_name = model_names[i]  # Get the model name using the index
            predicted_fares.append((model_name, predicted_fare[0]))

    # Display predicted fares after loading spinner completes
    st.write("Predicted Fares:")
    for model_name, fare in predicted_fares:
        st.write(f"{model_name} Model: ${fare:.2f}")
