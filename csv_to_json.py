import pandas as pd
import json

def csv_to_json(csv_file, json_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert the DataFrame to a list of dictionaries (records)
    data = df.to_dict(orient='records')
    
    # Write the data to a JSON file
    with open(json_file, 'w') as json_f:
        json.dump(data, json_f, indent=4)

# Define the input CSV file and output JSON file
csv_file = 'fashion.csv'  # Your CSV file
json_file = 'products.json'  # Desired JSON file

# Call the function to convert CSV to JSON
csv_to_json(csv_file, json_file)

print("CSV to JSON conversion completed successfully.")
