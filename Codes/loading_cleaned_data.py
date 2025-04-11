import h5py


if __name__ == "__main__":

    base_directory = r'C:\Users\USER\Desktop\Extending the thesis\Dataset\BCICIV_2a_gdf_2'
    
    filename = r"EEG_python_ready_A01T"
    file_path = fr'{base_directory}\{filename}.mat'
    
    # Load training data
    with h5py.File(file_path, 'r') as file:
        X_train = file['X'][:]
        y_train = file['y'][:].flatten()
    
    filename = r"EEG_python_ready_A01E"
    file_path = fr'{base_directory}\{filename}.mat'
    
    # Load test  data
    with h5py.File(file_path, 'r') as file:
        X_test = file['X'][:]
        y_test = file['y'][:].flatten()

