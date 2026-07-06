import os
import glob
import json
import requests
import datetime
import h5py
import cftime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uptide

# ==========================================
# CONFIGURATION
# ==========================================
START_DATE = "2022-01-01T00:00:00Z"
END_DATE = "2022-02-15T00:00:00Z"
CONSTITUENTS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1']
DATASET_ID = "global_hourly_fast"  # 'global_hourly_rqds' for research quality (slower updates)

# Directories
OBS_DIR = "observations"
OUT_DIR = "processed_results"
MODEL_DIR = "outputs_spinup"
os.makedirs(OBS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

COORD_FILE = os.path.join(OBS_DIR, "uhslc_coordinates.csv")

# ==========================================
# 2. DOWNLOAD OBSERVATIONS (UHSLC)
# ==========================================
def download_station_data(station_id, start_date, end_date):
    """Downloads hourly sea level data for a specific station and date range."""
    csv_file = os.path.join(OBS_DIR, f"station_{station_id}_obs.csv")
    
    if os.path.exists(csv_file):
        print(f"  -> Station {station_id} data already downloaded.")
        # FIX: Change 'time (UTC)' to 'time'
        return pd.read_csv(csv_file, parse_dates=['time'], skiprows=[1]) 
    
    print(f"  -> Downloading Station {station_id} from UHSLC...")
    
    # Build ERDDAP URL
    url = (f"https://uhslc.soest.hawaii.edu/erddap/tabledap/{DATASET_ID}.csv"
           f"?station_name%2Ctime%2csea_level"
           f"&station_name=%22{station_id}%22"
           f"&time>={start_date}&time<={end_date}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(csv_file, "w") as f:
            f.write(response.text)
        # FIX: Change 'time (UTC)' to 'time'
        return pd.read_csv(csv_file, parse_dates=['time'], skiprows=[1])
    except requests.exceptions.RequestException as e:
        print(f"     Failed to download station {station_id}. It may not exist in this timeframe.")
        return None

# ==========================================
# 3. UPTIDE HARMONIC ANALYSIS
# ==========================================
def get_harmonics(station_id, df_obs):
    """Calculates or loads previously calculated harmonic constituents."""
    json_file = os.path.join(OBS_DIR, f"station_{station_id}_constituents.json")
    
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        print(f"  -> Loaded stored constituents for station {station_id}.")
        return data['amplitudes'], data['phases']
    
    print(f"  -> Calculating constituents for station {station_id} via uptide...")
    
    # Clean data: drop NaNs using the corrected column name 'sea_level'
    df_clean = df_obs.dropna(subset=['sea_level']).copy()
    if df_clean.empty:
        return None, None

    tides = uptide.Tides(CONSTITUENTS)
    # Use 'time' instead of 'time (UTC)'
    ref_time = df_clean['time'].iloc[0].to_pydatetime()
    tides.set_initial_time(ref_time)
    
    # Convert times to seconds since reference time
    t_seconds = (df_clean['time'] - df_clean['time'].iloc[0]).dt.total_seconds().values
    
    # CRITICAL: Convert millimeters to meters for your model comparison!
    elev_vals = df_clean['sea_level'].values / 1000.0 
    
    # Perform harmonic analysis
    amp, pha = uptide.harmonic_analysis(tides, elev_vals, t_seconds)
    
    # Store for later
    harmonics_data = {
        'reference_time': ref_time.isoformat(),
        'amplitudes': list(amp),
        'phases': list(pha),
        'constituents': CONSTITUENTS
    }
    with open(json_file, "w") as f:
        json.dump(harmonics_data, f, indent=4)
        
    return list(amp), list(pha)

# ==========================================
# 4. LOAD MODEL AND COMPARE
# ==========================================
def process_station(model_file):
    """Orchestrates data loading, harmonic prediction, and plotting for a single file."""
    # Extract station name from filename: outputs/diagnostic_timeseries_STATION_elev.hdf5
    basename = os.path.basename(model_file)
    sta = basename.replace("diagnostic_timeseries_", "").replace("_elev.hdf5", "")
    print(f"\nProcessing Station: {sta}")
    
    # 1. Download observation data
    df_obs = download_station_data(sta, START_DATE, END_DATE)
    if df_obs is None or df_obs.empty:
        print(f"  -> Skipping {sta}: No observation data.")
        return

    # 2. Get harmonics
    amp, pha = get_harmonics(sta, df_obs)
    if amp is None:
        print(f"  -> Skipping {sta}: Could not calculate harmonics.")
        return
        
    # 3. Load Model Data
    print(f"  -> Loading model data...")
    with h5py.File(model_file, "r") as h5file:
        time_raw = h5file["time"][:].flatten()
        time_units = h5file["time"].attrs['units']
        # Convert cftime to standard datetime objects for uptide compatibility
        cf_time = cftime.num2pydate(time_raw, time_units)
        model_time = np.array([datetime.datetime(*t.timetuple()[:6]) for t in cf_time])
        model_elev = h5file["elev"][:].flatten()

    # 4. Predict Tides at exact model timesteps
    print(f"  -> Predicting tides aligned with model timestamps...")
    
    # Read the reference time used during the harmonic analysis
    json_file = os.path.join(OBS_DIR, f"station_{sta}_constituents.json")
    with open(json_file, "r") as f:
        ref_time_str = json.load(f)['reference_time']
        ref_time = datetime.datetime.fromisoformat(ref_time_str).replace(tzinfo=None)

    tides = uptide.Tides(CONSTITUENTS)
    tides.set_initial_time(ref_time)
    
    # Convert model times into seconds since the observation reference time
    t_predict_seconds = np.array([(mt - ref_time).total_seconds() for mt in model_time])
    
    # Predict tide based on calculated amplitude and phase
    pred_elev = tides.from_amplitude_phase(amp, pha, t_predict_seconds)
    
    # 5. Save Combined Data
    out_df = pd.DataFrame({
        'time': model_time,
        'model_elev': model_elev,
        'predicted_tide': pred_elev
    })
    out_csv = os.path.join(OUT_DIR, f"{sta}_comparison.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"  -> Saved combined data to {out_csv}")

    # 6. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(out_df['time'], out_df['predicted_tide'], label='UHSLC Predicted Tide (uptide)', color='blue', alpha=0.7)
    plt.plot(out_df['time'], out_df['model_elev'], label='Model Elevation', color='red', linestyle='--', alpha=0.8)
    plt.title(f"Tidal Comparison: Model vs Harmonic Prediction - Station {sta}")
    plt.xlabel("Date")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(OUT_DIR, f"{sta}_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"  -> Saved plot to {plot_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Find all model files in the target directory
    model_files = glob.glob(os.path.join(MODEL_DIR, "diagnostic_timeseries_*_elev.hdf5"))
    
    if not model_files:
        print(f"No model HDF5 files found in '{MODEL_DIR}'.")
    else:
        for m_file in model_files:
            process_station(m_file)
            
    print("\nProcessing complete.")
