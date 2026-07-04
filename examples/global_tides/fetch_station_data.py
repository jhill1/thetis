import pandas as pd

def fetch_uhslc_stations():
    """
    Fetches unique station names and coordinates from the UHSLC ERDDAP server
    and saves them to a local CSV file.
    """
    # We use the Research Quality dataset (rqds) as it contains the most 
    # comprehensive list of historical and active stations. 
    # If you only want currently active stations, change this to 'global_hourly_fast'
    dataset_id = "global_hourly_rqds"
    
    # Construct the ERDDAP URL
    # We request the variables station_name, latitude, and longitude.
    # The &distinct() parameter ensures we only download the metadata table.
    url = f"https://uhslc.soest.hawaii.edu/erddap/tabledap/{dataset_id}.csv?station_name,latitude,longitude&distinct()"
    
    print(f"Fetching tide gauge metadata from UHSLC...")
    
    try:
        # Load the CSV data directly from the URL into a pandas DataFrame.
        # ERDDAP always returns a second row containing unit strings 
        # (e.g., 'degrees_north', 'degrees_east'). We skip it using skiprows=[1].
        df = pd.read_csv(url, skiprows=[1])
        
        # Clean up the column names for easier reading
        df.rename(columns={
            'station_name': 'Station Name',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        }, inplace=True)
        
        # Save the DataFrame to a CSV file
        output_filename = "uhslc_station_locations.csv"
        df.to_csv(output_filename, index=False)
        
        print(f"Success! Saved {len(df)} stations to '{output_filename}'.")
        
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")

if __name__ == "__main__":
    fetch_uhslc_stations()
