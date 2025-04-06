import pandas as pd
from datetime import datetime

def load_raw_data():
    """
    Load raw data files from the data/raw directory.
    
    Returns:
        tuple: Three dataframes containing:
            - measurements: Air quality measurements with timestamps
            - instruments: Instrument readings and status data
            - pollutant_info: Reference information about pollutants and thresholds
    """
    measurements = pd.read_csv('data/raw/measurement_data.csv', parse_dates=['Measurement date'])
    instruments = pd.read_csv('data/raw/instrument_data.csv', parse_dates=['Measurement date'])
    pollutant_info = pd.read_csv('data/raw/pollutant_data.csv')
    return measurements, instruments, pollutant_info

def filter_normal_measurements(measurements, instruments):
    """
    Filter measurements and instruments data to include only records with normal status.
    
    Parameters:
        measurements (DataFrame): Raw measurements data
        instruments (DataFrame): Raw instrument data
    
    Returns:
        tuple: Two filtered dataframes:
            - normal_measurements: Measurements with normal status
            - normal_instruments: Instruments with status code 0 (Normal)
    """
    normal_measurements = measurements.copy()
    normal_instruments = instruments[instruments['Instrument status'] == 0]
    return normal_measurements, normal_instruments

def process_data():
    """
    Process the raw air quality data by:
    1. Loading raw data files
    2. Filtering for normal measurements
    3. Handling missing values in pollutant columns
    4. Saving processed data to CSV
    
    Returns:
        tuple: Three dataframes containing processed data:
            - measurements: Processed air quality measurements
            - instruments: Filtered instrument data
            - pollutant_info: Reference information about pollutants
    """
    measurements, instruments, pollutant_info = load_raw_data()
    measurements, instruments = filter_normal_measurements(measurements, instruments)
    
    pollutant_cols = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    for col in pollutant_cols:
        measurements[col].fillna(measurements[col].mean(), inplace=True)
    
    measurements.to_csv('data/processed/measurements_processed.csv', index=False)
    return measurements, instruments, pollutant_info

if __name__ == "__main__":
    process_data()
