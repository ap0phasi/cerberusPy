
import pandas as pd
from typing import List, Tuple
import numpy as np

# For CUDA acceleration
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

from torch.utils.data import DataLoader, TensorDataset
import torch
import random

def downsample_timeseries_data(df: pd.DataFrame, 
                                feature_indexes: dict,
                                window_timesteps: dict
                                ) -> dict:
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

    for key in feature_indexes:
        processed_data[key] = downsample(df.iloc[:,feature_indexes[key]], window_timesteps[key])

    return processed_data


def slice_timeseries_data(data_to_slice: pd.DataFrame,
                          downsampled_data: pd.DataFrame, 
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
    
    sliced_data = {}
    for key in data_to_slice:
        sliced_data[key] = []
    # To store our last known unnormalized value
    sliced_data['last_known'] = []
    
    selected_timestamps = []
    # Loop through each timestamp in your data_to_slice
    for timestamp in data_to_slice['response'].index:
        
        # Check if we have enough, call, context, and response data. 
        check_size_dict={}
        for key in data_to_slice:
            if key == 'response':
                check_size_dict[key] = get_following_timestamps(timestamp, data_to_slice[key], sizes[key])
            else:
                check_size_dict[key] = get_preceding_timestamps(timestamp, data_to_slice[key], sizes[key])
            
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
            # Include last known value (pre-normalization) from call
            sliced_data['last_known'].append(get_preceding_timestamps(timestamp, downsampled_data['call'], 1))
            
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
            
    unmasked_response = np.repeat(sliced_data['response'],sizes['response'],axis=0)
            
    return expanded_dict, response_data, unmasked_response

def coil_normalization(df):
    def max_absolute_change(df_diff):
        # Calculate the absolute change for each feature
        abs_change = df_diff.abs()

        # Find the maximum absolute change for each feature
        max_changes = abs_change.max()

        # Convert the series to a DataFrame
        max_change_df = max_changes.to_frame(name='Max Absolute Change')

        return max_change_df
    
    # Calculate the change for each feature
    df_diff = df.diff()
    # Calculate the max absolute change for each feature
    max_change_df = max_absolute_change(df_diff)
    
    normalized_df = pd.DataFrame()

    for column in df_diff.columns:
        max_change_val = max_change_df.loc[column,'Max Absolute Change']
        normalized_df[column] = (df_diff[column] + max_change_val) / (2 * max_change_val)
    # Set first row to 0 as there is no diff
    normalized_df.iloc[0] = normalized_df.iloc[0].fillna(0)
       
    return normalized_df, max_change_df


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

# Make Tensor for coil-normalized data
def make_torch_tensor(channel_array):
    first_channel = torch.tensor(channel_array, dtype=torch.float32)

    # Create the second channel as 1 minus the first channel
    second_channel = 1 - first_channel

    # Combine both channels to form a two-channel tensor
    # The unsqueeze(1) adds a channel dimension
    both_channels = torch.stack((first_channel, second_channel), dim=1)
    
    return both_channels

def prepare_timeseries_data(df,
                            sizes,
                            thresholds,
                            feature_indexes,
                            window_timesteps,
                            train_len,
                            feature_range = [0, 1]):
    
    # Calculate the min-max dataframe
    min_max_df = create_min_max_df(df)
    # Perform min-max scaling to specified feature range
    scaled_df = scale_data(df, min_max_df, feature_range)
    
    # Downsample data based on window timesteps
    downsampled_data = downsample_timeseries_data(scaled_df, 
                                        feature_indexes,
                                        window_timesteps)
    
    # Perform Coil Normalization
    normalized_data = {}
    max_change_dfs = {}
    for key in downsampled_data:
        normalized_data[key], max_change_dfs[key] = coil_normalization(downsampled_data[key])
        
    # Perform data slicing
    sliced_data, selected_timestamps = slice_timeseries_data(normalized_data,
                                                        downsampled_data,
                                                        sizes,
                                                        thresholds)
    
    # Perform masked expansion
    expanded_dict, response_data, unmasked_response = masked_expand(sliced_data, sizes)
    
    # Randomly sample training data
    train_index = random.sample(range(train_len), train_len)

    # We will convert our numpy arrays to 2-channel tensors
    calls = make_torch_tensor(expanded_dict['call'][train_index,:,:])
    contexts = [make_torch_tensor(expanded_dict[key][train_index,:,:]) for key in expanded_dict if 'context' in key]
    responses = make_torch_tensor(expanded_dict['response'][train_index,:,:])

    # Last knowns isn't coil-normalized so we won't process it as such
    last_knowns = torch.tensor(expanded_dict['last_known'][train_index,0,:], dtype=torch.float32)

    # We don't need to produce both channels of the coil normalization, we can just do the first. 
    y = torch.tensor(response_data[train_index,:], dtype=torch.float32)

    # Create separate datasets for each context
    datasets = [TensorDataset(calls, context, responses, last_knowns, y) for context in contexts]

    # Create a DataLoader for each dataset
    dataloaders = [DataLoader(dataset, batch_size=100, shuffle=True) for dataset in datasets]
    
    # Prepare each DataLoader
    prepared_dataloaders = [accelerator.prepare(dataloader) for dataloader in dataloaders]
    
    return prepared_dataloaders, sliced_data


def denormalize(normalized_df, max_change_df, initial_value, start_idx, end_idx):
    # Initialize the reconstructed DataFrame with the initial value
    reconstructed_df = pd.DataFrame(index=normalized_df.index[start_idx-1:end_idx])
    for column in normalized_df.columns:
        reconstructed_df[column] = initial_value[column]
        # Iteratively reconstruct the signal
        for i in range(start_idx, end_idx):
            delta_max = max_change_df.loc[column, 'Max Absolute Change']
            delta_t = 2 * normalized_df.at[normalized_df.index[i], column] * delta_max - delta_max
            reconstructed_df.at[normalized_df.index[i], column] = reconstructed_df.at[normalized_df.index[i - 1], column] + delta_t
    reconstructed_df.drop(index=reconstructed_df.index[0], axis=0, inplace=True)
    return reconstructed_df

def generate_predictions(model,selected_data):
    calls = make_torch_tensor(selected_data['call'])
    contexts = [make_torch_tensor(selected_data[key]) for key in selected_data if 'context' in key]
    responses = make_torch_tensor(np.zeros([1,selected_data['response'].shape[1],selected_data['response'].shape[2]]))

    # Last knowns isn't coil-normalized so we won't process it as such
    last_knowns = torch.tensor(selected_data['last_known'][:,0,:], dtype=torch.float32)

    respones_generated = []
    for igen in range(responses.shape[2]):
        with torch.no_grad():
            res_out = model(calls, contexts, responses, last_knowns)
            responses[0,0,igen,:] = res_out[0]
            respones_generated.append(res_out[0].numpy())
        
    return np.vstack(respones_generated)