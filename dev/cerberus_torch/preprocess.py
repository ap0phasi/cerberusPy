import pandas as pd
from typing import List, Tuple
import numpy as np

def downsample_timeseries_data(df: pd.DataFrame, 
                            context_windows: List[str], 
                            call_window: str, 
                            response_window: str,
                            call_feature_index: List[int],
                            context_feature_index: List[List[int]],
                            response_feature_index: List[int]) -> dict:
    """
    Process a timeseries DataFrame based on user-defined parameters.

    Args:
    df (pd.DataFrame): The original DataFrame with a DateTimeIndex.
    time_windows (List[str]): List of time windows for downsampling (e.g., ['1H', '2D', '3W']).
    context_sizes (List[int]): List of context sizes corresponding to each time window.
    call_sizes (List[int]): List of call sizes for constructing lookback blocks.
    response_size (int): Size of the response block.

    Returns:
    dict: A dictionary containing the downsampled timeseries for each time window. 
    """
    processed_data = {}

    # Function to downsample the DataFrame
    def downsample(df, window):
        """
        Downsample the DataFrame, taking the mean of each window and setting the 
        index to the maximum timestamp in that window.
        """
        # First, calculate the mean for each window
        mean_df = df.resample(window).mean()

        # Then, find the maximum timestamp in each window
        max_timestamps = df.iloc[:,1].resample(window).apply(lambda x: x.index.max())

        # Set the maximum timestamps as the new index
        mean_df.index = max_timestamps.values

        return mean_df

    processed_data['call'] = downsample(df.iloc[:,call_feature_index], call_window)
    processed_data['response'] = downsample(df.iloc[:,response_feature_index], response_window)
    
    contexts = []
    for i, context_window in enumerate(context_windows):
        contexts.append(downsample(df.iloc[:,context_feature_index[i]], context_window))
        
    processed_data['contexts'] = contexts
    return processed_data

def slice_timeseries_data(downsampled_data: pd.DataFrame, 
                            sizes: dict,
                            thresholds: dict) -> np.array:
    
    def get_max_timestamp_less_than(timestamp, df_sel):
    # Find the maximum timestamp less than the given timestamp
        filtered_timestamps = df_sel.index[df_sel.index < timestamp]
        return filtered_timestamps.max() if not filtered_timestamps.empty else None

    def get_preceding_timestamps(timestamp, df_sel, slice_size):
        """
        Retrieve a specified number of timestamps preceding (and excluding) the given timestamp.
        """
        # Find the maximum timestamp less than the given timestamp
        max_less_than_timestamp = get_max_timestamp_less_than(timestamp, df_sel)
        if max_less_than_timestamp is None:
            return pd.DataFrame()  # Return empty DataFrame if no suitable timestamp is found

        # Find the index of this timestamp in df_sel
        closest_idx = df_sel.index.searchsorted(timestamp, side='left')

        # Extract the preceding timestamps
        start_idx = max(closest_idx - slice_size, 0)
        relevant_timestamps = df_sel.iloc[start_idx:closest_idx, :]
        
        return relevant_timestamps  
    
    def get_following_timestamps(timestamp, df_sel, slice_size):
        """
        Retrieve timestamps equal to or greater than the selected timestamp.
        """
        # Find the index of the closest timestamp in df_sel
        closest_idx = df_sel.index.searchsorted(timestamp, side='left')
        
        # Ensure index is within bounds
        if closest_idx >= len(df_sel):
            return pd.DataFrame()  # Return empty DataFrame if index is out of bounds

        # Extract the following timestamps
        end_idx = min(closest_idx + slice_size, len(df_sel))
        relevant_timestamps = df_sel.iloc[closest_idx:end_idx, :]
        
        return relevant_timestamps
    
    def fill_numpy(relevant_timestamps,slice_size, direction):
        # Create a placeholder array
        placeholder_array = np.zeros([slice_size, relevant_timestamps.shape[1]])
        
        # Fill the placeholder array from the bottom
        if direction == "preceding":
            placeholder_array[-len(relevant_timestamps):] = relevant_timestamps.to_numpy()
        else:
            placeholder_array[:len(relevant_timestamps)] = relevant_timestamps.to_numpy()
        
        return placeholder_array
    
    context_sizes = [sizes[key] for key in sizes if 'context' in key]
    sliced_data = {f'context_{i}': [] for i in range(len(context_sizes))}
    sliced_data['call'] = []
    sliced_data['response'] = []
    
    selected_timestamps = []
    # Loop through each timestamp in your downsampled_data
    for timestamp in downsampled_data['response'].index:
        # Check if we have enough, call, context, and response data. 
        check_size_dict = {f'context_{i}': get_preceding_timestamps(timestamp, downsampled_data['contexts'][i], sizes[f'context_{i}']) for i in range(len(context_sizes))}
        check_size_dict['call'] = get_preceding_timestamps(timestamp, downsampled_data['call'], sizes['call'])
        check_size_dict['response'] = get_following_timestamps(timestamp, downsampled_data['response'], sizes['response'])
        
        size_check = []
        for key, value in check_size_dict.items():
            size_check.append((value.shape[0] / sizes[key]) >= thresholds[key])

        if size_check.count(False)==0:
            # Add timestamp to list
            selected_timestamps.append(timestamp)
    
            for key in check_size_dict:
                if key == 'response':
                    direction = 'following'
                else:
                    direction = 'preceding'
                # Append the result of get_preceding_timestamps to the list in the dictionary
                sliced_data[key].append(fill_numpy(check_size_dict[key], sizes[key], direction))
            
    # After the loop, use np.vstack to combine the arrays for each key
    stacked_arrays_dict = {key: np.stack(value) for key, value in sliced_data.items()}
    
    return stacked_arrays_dict, selected_timestamps

def masked_expand(sliced_data, sizes):
    sizes['response']
    
    expanded_dict = {}
    for key, value in sliced_data.items():
        if key == "response":
            res_shape = value.shape
            response_data = value.reshape(res_shape[0]*res_shape[1],res_shape[2])
            mask_array = np.tile(np.repeat(np.tril(np.ones([res_shape[1],res_shape[1]]),-1)[:,:,np.newaxis], res_shape[2], axis=2), (res_shape[0], 1, 1))
            expanded_dict[key] = np.repeat(value,sizes['response'],axis=0) * mask_array
        else:
            expanded_dict[key] = np.repeat(value,sizes['response'],axis=0)
            
    return expanded_dict, response_data

def create_min_max_df(df):
    """Create a DataFrame with min and max values for each feature."""
    min_values = df.min()
    max_values = df.max()
    min_max_df = pd.DataFrame({'min': min_values, 'max': max_values})
    return min_max_df

def scale_data(df, min_max_df, feature_range=(0, 1)):
    """Scale data in df based on min-max values and a given feature range."""
    min_scale, max_scale = feature_range
    scaled_df = pd.DataFrame()

    for column in df.columns:
        min_val = min_max_df.loc[column, 'min']
        max_val = min_max_df.loc[column, 'max']
        # Avoid division by zero in case of constant columns
        if max_val != min_val:
            scaled_df[column] = (df[column] - min_val) / (max_val - min_val) * (max_scale - min_scale) + min_scale
        else:
            scaled_df[column] = df[column]

    return scaled_df

if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv(r"data/jena_climate_2009_2016.csv",
                    parse_dates=['Date Time'], 
                    index_col=['Date Time'])
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')
    df = df.iloc[:10000,:]
    context_windows = ['1H', '2H', '6H']
    context_sizes = [24, 12, 6]
    call_window = '10T'
    call_size = 24
    response_window = '10T'
    response_size = 8
    call_feature_index = range(0,14)
    context_feature_index = [range(0,14),
                            range(0,14),
                            range(0,14)]
    response_feature_index = [0, 4, 12]
    thresholds = {
        'call': 0.7,
        'response': 0.7,
        'context_0': 0.7,
        'context_1': 0.7,
        'context_2': 0.7
    }
    sizes = {
        'call': 24,
        'response': 8,
        'context_0': 24,
        'context_1': 12,
        'context_2': 6
    }

    #Scale Data
    min_max_df = create_min_max_df(df)
    print(min_max_df)
    scaled_df = scale_data(df, min_max_df, feature_range=(0, 1))

    downsampled_data = downsample_timeseries_data(scaled_df, 
                                            context_windows, 
                                            call_window, 
                                            response_window,
                                            call_feature_index,
                                            context_feature_index,
                                            response_feature_index)
    sliced_data, selected_timestamps = slice_timeseries_data(downsampled_data,
                                        sizes,
                                        thresholds)

    for key in sliced_data:
        print(sliced_data[key].shape)

    expanded_dict, response_data = masked_expand(sliced_data, sizes)

    print(response_data[0,:])
    for ir in range(30):
        print(expanded_dict['response'][ir,:,:])
        
    for key in expanded_dict:
        print(expanded_dict[key].shape)
        
    print(response_data.shape)
