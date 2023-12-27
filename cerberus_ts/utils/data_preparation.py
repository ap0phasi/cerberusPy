
import pandas as pd
from typing import List, Tuple
import numpy as np

# For CUDA acceleration
from accelerate import Accelerator

from torch.utils.data import DataLoader, TensorDataset
import torch
import random

import matplotlib.pyplot as plt

# Import configuration
from .cerberus_config import CerberusConfig

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
    
    if CerberusConfig.set_masked_norm_zero:
        # We want to make sure we maintain distinction between a response that is the maximum decrease and one that is masked.
        # In the case where the first channel is 0, we will assume this denotes a NaN, so we will replace it like this:
        second_channel[first_channel == 0] = 0

    # Combine both channels to form a two-channel tensor
    # The unsqueeze(1) adds a channel dimension
    both_channels = torch.stack((first_channel, second_channel), dim=1)
    
    return both_channels

def denormalize_response(normalized_df, max_change_df, initial_value):
    reconstructed_array = []
    new_value = initial_value
    reconstructed_array.append(initial_value)
    delta_max = np.array(max_change_df).T
    for i in range(normalized_df.shape[0]):
        delta_t = 2 * normalized_df[i,:] * delta_max - delta_max
        new_value = new_value + delta_t
        reconstructed_array.append(new_value[0])
        
    return pd.DataFrame(reconstructed_array)

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
            # For multi-channel coil-normalized heads
            responses[0,1,igen,:] = 1 - res_out[0]
            
            if CerberusConfig.set_masked_norm_zero:
                # In the case where the first channel is 0, we will assume this denotes a NaN, so we will replace it like this:
                responses[0,1,igen,:][responses[0,0,igen,:] == 0] = 0
            
            respones_generated.append(res_out[0].numpy())
        
    return np.vstack(respones_generated)
def invert_scaling(scaled_array, min_max_df, feature_range=(0, 1)):
    """
    Inverts the scaling for a numpy array based on min-max values and a given feature range.

    :param scaled_array: numpy array of scaled data.
    :param min_max_df: DataFrame with 'min' and 'max' values for each feature.
    :param feature_range: tuple indicating the range used during scaling.
    :return: DataFrame of original scale data.
    """
    min_scale, max_scale = feature_range
    original_data = np.zeros_like(scaled_array)

    for i in range(scaled_array.shape[1]):
        min_val = min_max_df.iloc[i]['min']
        max_val = min_max_df.iloc[i]['max']
        # Reverse the scaling formula
        original_data[:, i] = ((scaled_array.iloc[:, i] - min_scale) / (max_scale - min_scale)) * (max_val - min_val) + min_val

    return pd.DataFrame(original_data)


class TimeseriesDataPreparer:
    def __init__(self, df, sizes, thresholds, feature_indexes, window_timesteps, train_len, feature_range=(0, 1), batch_size = 100):
        self.df = df
        self.sizes = sizes
        self.thresholds = thresholds
        self.feature_indexes = feature_indexes
        self.window_timesteps = window_timesteps
        self.train_len = train_len
        self.feature_range = feature_range
        self.batch_size = batch_size

        # Initialize attributes to store results
        self.min_max_df = None
        self.scaled_df = None
        self.downsampled_data = None
        self.normalized_data = None
        self.max_change_dfs = None
        self.sliced_data = None
        self.selected_timestamps = None
        self.expanded_dict = None
        self.response_data = None
        self.unmasked_response = None
        self.dataloaders = None

    def prepare_data(self):
        # Perform all the steps of data preparation
        self.min_max_df = create_min_max_df(self.df)
        self.scaled_df = scale_data(self.df, self.min_max_df, self.feature_range)
        self.downsampled_data = downsample_timeseries_data(self.scaled_df, self.feature_indexes, self.window_timesteps)

        self.normalized_data, self.max_change_dfs = {}, {}
        for key in self.downsampled_data:
            self.normalized_data[key], self.max_change_dfs[key] = coil_normalization(self.downsampled_data[key])

        self.sliced_data, self.selected_timestamps = slice_timeseries_data(self.normalized_data, self.downsampled_data, self.sizes, self.thresholds)
        self.expanded_dict, self.response_data, self.unmasked_response = masked_expand(self.sliced_data, self.sizes)

        train_index = random.sample(range(self.train_len), self.train_len)
        calls = make_torch_tensor(self.expanded_dict['call'][train_index, :, :])
        contexts = [make_torch_tensor(self.expanded_dict[key][train_index, :, :]) for key in self.expanded_dict if 'context' in key]
        responses = make_torch_tensor(self.expanded_dict['response'][train_index, :, :])
        last_knowns = torch.tensor(self.expanded_dict['last_known'][train_index, 0, :], dtype=torch.float32)
        y = torch.tensor(self.response_data[train_index, :], dtype=torch.float32)
        # For foresight we also need the unmasked response prepared
        unmasked = torch.tensor(self.unmasked_response[train_index, :, :], dtype=torch.float32)

        datasets = [TensorDataset(calls, context, responses, last_knowns, y, unmasked) for context in contexts]
        self.dataloaders = [DataLoader(dataset, self.batch_size, shuffle=True) for dataset in datasets]

        # For CUDA Acceleration
        accelerator = Accelerator()
        self.dataloaders = [accelerator.prepare(dataloader) for dataloader in self.dataloaders]

class ResponseGenerator:
    def __init__(self, model, sliced_data, feature_indexes, max_change_dfs):
        self.model = model
        self.sliced_data = sliced_data
        self.feature_indexes = feature_indexes
        self.max_change_dfs = max_change_dfs
        
        # Initialize attributes to store results
        self.responses_generated = None
        self.denormalized_response = None
        self.selected_data = None
        self.observed_response = None
        self.observed_unscaled = None
        self.modeled_unscaled = None

    def generate_response(self, sel_index):
        # Move model to CPU and set to evaluation mode
        self.model.to("cpu")
        self.model.eval()

        # Select data for the specified index
        self.selected_data = {key: value[sel_index:sel_index+1, :] for key, value in self.sliced_data.items()}
        self.responses_generated = generate_predictions(self.model, self.selected_data)
        
        # Find last known values to condition denormalizaiton
        initial_value = self.selected_data['last_known'][0][0][self.feature_indexes['response']]
        max_change_df = self.max_change_dfs['response']

        # Denormalize the response
        self.denormalized_response = denormalize_response(self.responses_generated, max_change_df, initial_value)
        self.observed_response = denormalize_response(self.selected_data['response'][0,:,:], max_change_df, initial_value)
        
    def unscale_response(self, min_max_df, feature_indexes):
        scale_frame = min_max_df.iloc[feature_indexes['response'],:]
        self.observed_unscaled = invert_scaling(self.observed_response, scale_frame, feature_range = (0,1))
        self.modeled_unscaled = invert_scaling(self.denormalized_response, scale_frame, feature_range = (0,1))
    
    def plot_normalized_responses(self):
        observed = self.selected_data['response'][0,:,:]
        modeled = self.responses_generated

        # Number of rows and columns
        num_rows, num_cols = observed.shape

        # Create a plot for each feature (column)
        for i in range(num_cols):
            plt.figure(figsize=(10, 6))
            plt.plot(observed[:, i], label='Observed Change - Feature {}'.format(i+1))
            plt.plot(modeled[:, i], label='Modeled Change - Feature {}'.format(i+1))
            plt.title(f'Feature {i+1} Comparison')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
            
    def plot_unscaled_responses(self, min_max_df, feature_indexes):
        self.unscale_response(min_max_df,feature_indexes)

        # Number of rows and columns
        num_rows, num_cols = self.observed_unscaled.shape

        # Create a plot for each feature (column)
        for i in range(num_cols):
            plt.figure(figsize=(10, 6))
            plt.plot(self.observed_unscaled.iloc[:, i], label='Observed Unscaled - Feature {}'.format(i+1))
            plt.plot(self.modeled_unscaled.iloc[:, i], label='Modeled Unscaled - Feature {}'.format(i+1))
            plt.title(f'Feature {i+1} Comparison')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
            
    def plot_denormalized_responses(self):
        # Example matrices
        observed = self.observed_response
        modeled = self.denormalized_response

        # Number of rows and columns
        num_rows, num_cols = observed.shape

        # Create a plot for each feature (column)
        for i in range(num_cols):
            plt.figure(figsize=(10, 6))
            plt.plot(observed.iloc[:, i], label='Observed Scaled - Feature {}'.format(i+1))
            plt.plot(modeled.iloc[:, i], label='Modeled Scaled - Feature {}'.format(i+1))
            plt.title(f'Feature {i+1} Comparison')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()