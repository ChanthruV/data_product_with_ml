import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# Function to preprocess user input
def preprocess_user_input(origin_airport, destination_airport, flight_date, search_date, is_basic_economy,
                          is_refundable, is_non_stop, cabin_type, airline):

    # Convert boolean inputs to integer (0 or 1)
    is_basic_economy = int(is_basic_economy)
    is_refundable = int(is_refundable)
    is_non_stop = int(is_non_stop)

    # Convert 'search_date' and 'flight_date' to datetime, then to UNIX timestamp
    search_date_timestamp = int(pd.to_datetime(search_date).timestamp())
    flight_date_timestamp = int(pd.to_datetime(flight_date).timestamp())

    # Specify the file path to the CSV data
    file_path = r'C:\Users\chant\adv_mla_2023\data_product_with_ml\data\processed\extracted_data.csv'

    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Group the data by starting and destination airports
    grouped = data.groupby(['startingAirport', 'destinationAirport'])

    # Calculate the average travel duration and total travel distance for each group
    average_travel_data = grouped[['travelDuration', 'totalTravelDistance']].mean().reset_index()

    # Find the matching row for the provided origin and destination airports
    matching_row = average_travel_data[(average_travel_data['startingAirport'] == origin_airport) &
                                       (average_travel_data['destinationAirport'] == destination_airport)]

    # Extract the average travel duration and total travel distance
    average_travel_duration = matching_row['travelDuration'].values[0]
    average_total_travel_distance = matching_row['totalTravelDistance'].values[0]

    # Initialize the hasher for categorical features
    hasher = FeatureHasher(n_features=4, input_type='string')

    # Create a dataframe to hold the input for hash encoding
    df_for_hashing = pd.DataFrame({
        'origin_airport': [origin_airport],
        'destination_airport': [destination_airport],
        'cabin_type': [cabin_type],
        'airline': [airline]
    })

    # Apply hash encoding and convert the result to an array
    hashed_features = hasher.transform(df_for_hashing.to_dict(orient='records')).toarray()

    # Flatten hashed features and combine with other processed inputs, including calculated values
    user_input = hashed_features.flatten().tolist()
    user_input.extend([search_date_timestamp, flight_date_timestamp, is_basic_economy, is_refundable, is_non_stop])
    user_input.extend([average_travel_duration, average_total_travel_distance])  # Add calculated values

    print(hashed_features.shape)
    print(user_input)

    return user_input



# Define the available options for starting and destination airports and airline names

starting_airports = [
    'ATL', 'BOS', 'CLT', 'DEN', 'DFW', 'DTW', 'EWR', 'IAD', 'JFK',
    'LAX', 'LGA', 'MIA', 'OAK', 'ORD', 'PHL', 'SFO'
]

destination_airports = [
    'BOS', 'CLT', 'DEN', 'DFW', 'DTW', 'EWR', 'IAD', 'JFK', 'LAX',
    'LGA', 'MIA', 'OAK', 'ORD', 'PHL', 'SFO', 'ATL'
]

# Unique airline names extracted and cleaned from the array provided

airline_names = [
    'Alaska Airlines', 'American Airlines', 'Boutique Air', 'Cape Air',
    'Contour Airlines', 'Delta', 'Frontier Airlines', 'Hawaiian Airlines',
    'JetBlue Airways', 'Key Lime Air', 'Southern Airways Express',
    'Spirit Airlines', 'Sun Country Airlines', 'United'
]


def get_starting_airports():
    return starting_airports

def get_destination_airports():
    return destination_airports

def get_airline_names():
    return airline_names



