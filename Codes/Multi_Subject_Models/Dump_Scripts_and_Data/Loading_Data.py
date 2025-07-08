import os
import h5py
import numpy as np
import pandas as pd
import re

def extract_hdf5_data(h5_object):
    """Extracts data from an HDF5 object, ensuring proper conversion."""
    if isinstance(h5_object, h5py.Dataset):
        return np.array(h5_object)
    elif isinstance(h5_object, h5py.Group):
        return {key: extract_hdf5_data(h5_object[key]) for key in h5_object.keys()}
    return h5_object

def resolve_hdf5_references(h5_file, ref_array):
    """Resolves HDF5 object references and retrieves their actual values."""
    resolved_values = []
    for ref in ref_array:
        if isinstance(ref, h5py.Reference):
            resolved_values.append(h5_file[ref][()][0] if h5_file[ref].shape else h5_file[ref][()])  # Extract value correctly
        else:
            resolved_values.append(ref)  # If it's already a value, append directly
    return np.array(resolved_values, dtype=float) if len(resolved_values) > 0 else np.array([])

def decode_event_types(event_type_list):
    """Decodes event type numerical values into readable strings."""
    decoded_types = []
    for event in event_type_list:
        if isinstance(event, (list, np.ndarray)):
            decoded_types.append("".join([chr(int(num)) for num in np.array(event).flatten() if 32 <= int(num) <= 126]))
        else:
            decoded_types.append(str(event))
    return decoded_types

# Set the folder path containing the .mat files
input_folder = r"C:\Users\USER\Desktop\Extending the thesis\Dataset\BCICIV_2a_gdf"  # Update with your folder path

# Define subject IDs and session types to load
subject_ids = ["01", "02"]  # Add the subject numbers you want to fetch
session_types = ["T"]  # Choose session types (T for training, E for evaluation)

# List all .mat files in the folder
mat_files = [f for f in os.listdir(input_folder) if f.endswith(".mat")]

# Filter files based on subject IDs and session types
filtered_files = [f for f in mat_files if re.match(r"A(\d+)([TE])\.mat", f) and 
                  re.match(r"A(\d+)([TE])\.mat", f).group(1) in subject_ids and 
                  re.match(r"A(\d+)([TE])\.mat", f).group(2) in session_types]

# Initialize storage lists
eeg_data_list = []
event_data_list = []

# Loop through each filtered .mat file
for mat_file in filtered_files:
    # Extract subject ID and session type from filename
    match = re.match(r"A(\d+)([TE])\.mat", mat_file)
    if match:
        subject_id = match.group(1)
        session_type = match.group(2)
    else:
        subject_id = "Unknown"
        session_type = "Unknown"
    
    # Load the .mat file (HDF5 format)
    mat_path = os.path.join(input_folder, mat_file)
    with h5py.File(mat_path, "r") as f:
        eeg_data = extract_hdf5_data(f["EEG"])

        # Extract EEG signals
        eeg_signals = np.array(eeg_data["data"])  # EEG signal matrix (channels x time)
        sampling_rate = float(np.array(eeg_data["srate"]).flatten()[0])
        channel_count = eeg_signals.shape[0]
        timepoints = eeg_signals.shape[1]
        
        # Extract events safely and resolve references
        event_data = eeg_data["event"]
        event_types = resolve_hdf5_references(f, np.array(event_data["type"])).flatten()
        event_latencies = resolve_hdf5_references(f, np.array(event_data["latency"])).flatten()
        event_durations = resolve_hdf5_references(f, np.array(event_data["duration"])).flatten() if "duration" in event_data else None

        # Ensure latency values are numeric and scaled properly
        event_latencies = event_latencies.astype(float) / sampling_rate if event_latencies.size > 0 else np.array([])
        if event_durations is not None and event_durations.size > 0:
            event_durations = event_durations.astype(float) / sampling_rate
        else:
            event_durations = None

        # Decode event types into readable strings
        event_types = decode_event_types(event_types)

        event_list = []
        for i in range(min(len(event_types), len(event_latencies))):
            event_details = {
                "Filename": mat_file,
                "Subject ID": subject_id,
                "Session Type": session_type,
                "Latency (s)": event_latencies[i],
                "Event Type": event_types[i],
                "Duration (s)": event_durations[i] if event_durations is not None and i < len(event_durations) else None,
            }
            event_list.append(event_details)

    # Store data
    eeg_data_list.append({
        "Filename": mat_file,
        "Subject ID": subject_id,
        "Session Type": session_type,
        "EEG Signals": eeg_signals,
        "Sampling Rate": sampling_rate,
        "Channels": channel_count,
        "Timepoints": timepoints
    })
    event_data_list.extend(event_list)

# Convert to DataFrames for easier visualization
eeg_df = pd.DataFrame(eeg_data_list)
event_df = pd.DataFrame(event_data_list)

# Display extracted data
import ace_tools as tools
tools.display_dataframe_to_user(name="EEG Data Overview", dataframe=eeg_df)
tools.display_dataframe_to_user(name="Event Data Overview", dataframe=event_df)
