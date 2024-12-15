from pype_schema.parse_json import JSONParser
from pype_schema.tag import VirtualTag
from define import RO_tag_mappping, RO_name_to_color, RO_item_to_color

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import f
from argparse import ArgumentParser
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore')
plt.rcParams['figure.dpi'] = 300
plt.rc('legend', fontsize=14) # using a size in points
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('figure', titlesize=18)

parser = ArgumentParser()
parser.add_argument('--tag_file', type=str, default='data/SB_data/data_info.csv', help='Path to the tag file')
parser.add_argument('--data_folder', type=str, default='data/SB_data/case_study_data', help='Path to the data folder')
parser.add_argument('--train_test_split', type=float, default=0.8, help='Train-test split ratio')
parser.add_argument('--data_plot', type=str, default="results/fault_detection/data.png", help='Path to save the data plot')
parser.add_argument('--json_path', type=str, default='json/Desal.json', help='Path to the network JSON file')
parser.add_argument('--T2Q_plot', type=str, default='results/fault_detection/T2Q_plot.png', help='Path to save the T2Q plot')
parser.add_argument('--PC_plot', type=str, default='results/fault_detection/PC_plot.png', help='Path to save the PC plot')
args = parser.parse_args()

class MeasurementData:
    def __init__(self, tag_file, data_folder, train_test_split=0.8, columns=None):
        self.data_folder = data_folder
        self.tag_file = tag_file
        self.train_test_split = train_test_split
        self.columns = columns
        self.data_info = pd.read_csv(self.tag_file)
        self.data = self.load_data()
        self.training_data = None
        self.testinv_data = None
        
    def load_data(self):
        '''
        Load the data according to the tag file
        The tag file format:
        Tag, unit, type, node_id
        '''
        data_files = os.listdir(self.data_folder)
        # Load all CSV files in the data folder
        sub_data = []
        for i, file in enumerate(data_files):
            if file.endswith('.csv'):
                sub_data.append(pd.read_csv(os.path.join(self.data_folder, file)))
        # Concatenate all data files
        data = pd.concat(sub_data)
        # data = pd.concat([pd.read_csv(os.path.join(self.data_folder, file)) for file in data_files])
        # remove "To date" column
        data.drop(columns=['To date'], inplace=True)
        # sort by "From date"
        data.sort_values(by='From date', inplace=True)
        # reset index and column names
        data.reset_index(drop=True, inplace=True)
        data.columns = [RO_tag_mappping[col] for col in data.columns]
        if self.columns:
            data = data[self.columns]
        print(f"Data loaded from {self.data_folder} with shape: {data.shape}")
        return data
    
    def print_data(self):
        print(self.data.head())

    def print_format(self):
        print(self.data_info)

    def plot_column(self, column):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data[column], label=column)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Time Series Plot of {column}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def filter_data(self, data, mode=1):
        # use intake flowrate > 500 to filter data
        # also return the index of the filtered data
        if mode == 0:
            data = data[self.data['intake flowrate (GPM)'] < 500]
            print(f"Filtered data with intake flowrate <= 500, new shape: {data.shape}")
        elif mode == 1:
            data_cp = data.copy()
            data_cp = data_cp[data_cp['intake flowrate (GPM)'] < 500]
            index = data_cp.index
            data_cp = data.copy()
            data_cp = data_cp[data_cp['intake flowrate (GPM)'] >= 500]
            print(f"Filtered data with intake flowrate > 500, new shape: {data_cp.shape}")
        return data_cp, index.tolist()

    def detect_and_clean_outliers(self, time_series, delta=0.3, K=1, bandwidth=13):
        """
        Detect and clean outliers in a univariate time series using an autoregressive model.

        Parameters:
            time_series (pd.Series): The raw time series data
            delta (float): Smoothing parameter [0.01, 0.3]
            K (float): Multiplicative factor 
            bandwidth (int): Bandwidth for the kernel smoother (default: 13)

        Returns:
            pd.Series: The cleaned and smoothed time series data.
        """
        # Initialize variables
        print(f'Detecting outliers with customerized algorithm')
        forecast = pd.Series(np.zeros_like(time_series), index=time_series.index)  # Forecast values
        sigma_e = pd.Series(np.zeros_like(time_series), index=time_series.index)   # Forecast standard deviation
        accepted_data = pd.Series(np.zeros_like(time_series), index=time_series.index)  # Cleaned data
        forecast.iloc[0] = time_series.iloc[0]  # Initialize first forecast

        delta_t = abs(time_series.iloc[1] - forecast.iloc[0]) if len(time_series) > 1 else 0

        for t in range(1, len(time_series)):
            forecast.iloc[t] = (1 - delta) * forecast.iloc[t - 1] + delta * time_series.iloc[t - 1]
            
            e_t = time_series.iloc[t] - forecast.iloc[t]
            delta_t = delta * abs(e_t) + (1 - delta) * delta_t
            sigma_e.iloc[t] = 1.25 * delta_t
            
            x_lower = forecast.iloc[t] - K * sigma_e.iloc[t]
            x_upper = forecast.iloc[t] + K * sigma_e.iloc[t]
            
            if time_series.iloc[t] < x_lower or time_series.iloc[t] > x_upper:
                accepted_data.iloc[t] = forecast.iloc[t]
            else:
                accepted_data.iloc[t] = time_series.iloc[t]

        smoothed_data = accepted_data.rolling(window=bandwidth, center=True).mean()
        return smoothed_data

    def preprocess_data(self, columns=None, fill_missing='drop', remove_outliers=0, num=None, filter_data=-1):
        '''
        Preprocess the data with the following options:
        - columns: Specify which columns to preprocess
        - fill_missing: Method to handle missing values
        - filter_data: Filter the data based on a condition 
            -1 for no filter
            0 for intake flowrate <= 500
            1 for intake flowrate > 500
        - remove_outliers: Remove outliers in training data using 
            1: the IQR method or 
            0: detect_and_clean_outliers function
            -1: no outlier removal
        - train_test_split: Split the data into training and test sets (e.g., 0.8 for 80% training data
        '''
        processed_data = self.data.copy()
        if num:
            processed_data = processed_data[:num]
        if columns:
            processed_data = processed_data[columns]
        
        # Handle missing values
        if fill_missing == 'ffill':
            processed_data.fillna(method='ffill', inplace=True)
        elif fill_missing == 'bfill':
            processed_data.fillna(method='bfill', inplace=True)
        elif fill_missing == 'mean':
            processed_data.fillna(processed_data.mean(), inplace=True)
        elif fill_missing == 'drop':
            processed_data.dropna(inplace=True)

        # update the data
        self.data = processed_data

        if self.train_test_split:
            m = len(processed_data)
            self.training_data = processed_data[:int(m * self.train_test_split)]
            self.testing_data = processed_data[int(m * self.train_test_split):]
            # reset index for testing data start from 0
            self.testing_data.reset_index(drop=True, inplace=True)
        
        # Filter data
        if filter_data != -1:
            # self.training_data = self.filter_data(self.training_data, filter_data)
            self.training_data_filtered, self.train_index = self.filter_data(self.training_data, filter_data)
            # self.training_data = self.filter_data(self.training_data, filter_data)
            self.testing_data_filtered, self.test_index = self.filter_data(self.testing_data, filter_data)

        # Remove outliers using the IQR method in train data
        if remove_outliers==1:
            for col in self.training_data_filtered.columns:
                if col == 'timestamp':
                    continue
                Q1 = self.training_data_filtered[col].quantile(0.25)
                Q3 = self.training_data_filtered[col].quantile(0.75)
                IQR = Q3 - Q1
                self.training_data_filtered = self.training_data_filtered[(self.training_data_filtered[col] >= Q1 - 1.5 * IQR) & (self.training_data_filtered[col] <= Q3 + 1.5 * IQR)]
        elif remove_outliers==0:
            for col in self.training_data_filtered.columns:
                if col == 'timestamp':
                    continue
                self.training_data_filtered[col] = self.detect_and_clean_outliers(self.training_data_filtered[col])
        
        # scale the data
        # self.training_data = self.auto_scale(self.training_data)
        # self.testing_data = self.auto_scale(self.testing_data)
        self.training_data_filtered = self.auto_scale(self.training_data_filtered)
        self.testing_data_filtered = self.auto_scale(self.testing_data_filtered)

        print(f'Preprocessed data with shape: {processed_data.shape}, train-test split: {self.train_test_split}')
        print(f'Filtered data with shape: {self.training_data_filtered.shape, self.testing_data_filtered.shape}')

    def auto_scale(self, data):
        '''
        Standardize the data (mean=0, variance=1) in all columns besides the timestamp
        '''
        scaler = StandardScaler()
        scaled_data = data.copy()
        columns = data.columns.tolist()
        columns.remove('timestamp')
        scaled_data[columns] = scaler.fit_transform(data[columns])
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
        return scaled_data

    def plot_type(self, data_type):
        '''
        Plot the data based on the data type
        '''
        data_columns = self.data_info[self.data_info['type'] == data_type]['Tag']
        data = self.data[data_columns]
        data.plot(subplots=True, figsize=(15, 10))
        plt.suptitle(f'Process Data for {data_type}')
        plt.show()  
    
    def plot_data(self, save=None, mode='test', shade=True, combined=True):
        '''
        Plot the testing data in a vertical stack with same x axis
        -  using the "timestamp" column as the x-axis
            YYYY-MM-DD HH:MM:SS -> MM/DD
        - set all y min to 0, max to the max value + 10% of the max value
        - add grid
        - shade the off hours (flowrate < 500)

        save: path to save the plot
        mode: 'all', 'train', 'test'
        combined: True, False, combine columns with same unit in one plot
        '''
        if mode == 'all':
            plot_data = self.data.copy()
        elif mode == 'train':
            plot_data = self.training_data.copy()
        elif mode == 'test':
            plot_data = self.testing_data.copy()
            # plot_data.reset_index(drop=True, inplace=True)

        print(f'Plotting {mode} data with shape: {plot_data.shape}')
        num_of_columns = len(plot_data.columns)
        timestamps = plot_data['timestamp']
        plot_data.drop(columns=['timestamp'], inplace=True)

        # collect the off hours in a list of pairs (start, end) in self.testing_data
        off_hours = []
        start = None
        end = self.test_index[0]
        for i in self.test_index:
            if not start:
                start = i
                end = i
                continue
            if (i-end) > 1:
                off_hours.append((start, end+1))
                start = i
            end = i
        off_hours.append((start, end+1))
        self.off_hours = off_hours

        if combined:
            fig, axs = plt.subplots(4, 1, figsize=(12, 6), sharex=True, gridspec_kw={'hspace': 0.2})
            # intake flowrate (GPM) and wastewater flowrate (GPM)
            axs[0].plot(plot_data['intake flowrate (GPM)'], label='Intake', color=RO_item_to_color['intake'], zorder=2)
            axs[0].plot(plot_data['wastewater flowrate (GPM)'], label='Wastewater', color=RO_item_to_color['wastewater'], zorder=2)
            if shade:
                for i, (start, end) in enumerate(off_hours):
                    axs[0].axvspan(start, end, color='gray', alpha=0.3, zorder=2)
            axs[0].set_ylabel('Flowrate\n(GPM)')
            axs[0].set_ylim(bottom=0, top=plot_data['wastewater flowrate (GPM)'].max() * 1.1)
            
            # intake conductivity (uS/cm)
            axs[1].plot(plot_data['intake conductivity (uS/cm)'], label='Intake', color=RO_item_to_color['intake'], zorder=1)
            if shade:
                for i, (start, end) in enumerate(off_hours):
                    axs[1].axvspan(start, end, color='gray', alpha=0.3)
            axs[1].set_ylabel('Conductivity\n(uS/cm)')
            axs[1].set_ylim(bottom=0, top=plot_data['intake conductivity (uS/cm)'].max() * 1.1)

            # HP Pump speed (RPM) and Circulation Pump speed (RPM)
            axs[2].plot(plot_data['HP Pump speed (RPM)'], label='HP Pump', color=RO_item_to_color['HP Pump'])
            axs[2].plot(plot_data['Circulation Pump speed (RPM)'], label='Circulation Pump', color=RO_item_to_color['Circulation Pump'])
            if shade:
                for i, (start, end) in enumerate(off_hours):
                    axs[2].axvspan(start, end, color='gray', alpha=0.3)
            axs[2].set_ylabel('Pump\nSpeed\n(RPM)')
            axs[2].set_ylim(bottom=0, top=plot_data['HP Pump speed (RPM)'].max() * 1.1)

            # HP Pump pressure (PSI) and Circulation Pump pressure (PSI)
            axs[3].plot(plot_data['HP Pump pressure (PSI)'], label='HP Pump', color=RO_item_to_color['HP Pump'])
            axs[3].plot(plot_data['Circulation Pump pressure (PSI)'], label='Circulation Pump', color=RO_item_to_color['Circulation Pump'])
            if shade:
                for i, (start, end) in enumerate(off_hours):
                    if i == 0:
                        axs[3].axvspan(start, end, color='gray', alpha=0.3, label='Off Hours')
                    else:
                        axs[3].axvspan(start, end, color='gray', alpha=0.3)
            axs[3].set_ylabel('Pump\nPressure\n(PSI)')
            axs[3].set_ylim(bottom=0, top=plot_data['HP Pump pressure (PSI)'].max() * 1.1)

            handles, labels = [], []
            for i in range(4):
                h, l = axs[i].get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            label_to_handle = dict(zip(labels, handles))
            axs[0].legend(label_to_handle.values(), label_to_handle.keys(), bbox_to_anchor=(0.85, 1.5), loc='upper left')
            
        else:
            plt.figure(figsize=(12, 8))
            axes = []
            for i, col in enumerate(plot_data):
                ax = plt.subplot(num_of_columns, 1, i+1)
                ax.plot(plot_data[col])
                if shade:
                    for start, end in off_hours:
                        ax.axvspan(start, end, color='gray', alpha=0.3, label='Off Hours')
                if i > 0:
                    ax.sharex(axes[0])
                else:
                    handles, labels = ax.get_legend_handles_labels()
                    label_to_handle = dict(zip(labels, handles))
                    ax.legend(label_to_handle.values(), label_to_handle.keys(), bbox_to_anchor=(1, 1), loc='upper left')

                ax.set_ylabel(col.replace(' ', '\n'))
                ax.set_ylim(bottom=0, top=plot_data[col].max() * 1.1)
                # add a vertical dashed line to separate training and testing data
                if mode=='all':
                    n_train = int(len(plot_data) * self.train_test_split)
                    ax.axvline(x=n_train, color='r', linestyle='--')
                axes.append(ax)

        # use the timestamp as x axis following the format MM/DD
        labels = timestamps.apply(lambda x: x[5:10])
        positions = range(0, len(labels)+1, len(labels)//10)
        plt.xticks(ticks=positions, labels=labels[::len(labels)//10])
        plt.xlabel('Timestamp')
        plt.suptitle(mode.capitalize() + ' Data')
        
        plt.tight_layout()

        if save:
            plt.savefig(save)
        else:
            plt.show()

class FaultDetectionSystem:
    def __init__(self, json_path, dataset, n_components=2, significance_level=0.01):
        """
        Initialize with training data to build a PCA model.

        Parameters:
        - training_data: DataFrame containing training samples for normal operation
        - n_components: Number of principal components to retain
        - significance_level: Significance level for threshold calculation
        """
        self.load_network(json_path)
        self.dataset = dataset
        self.n_components = n_components
        self.significance_level = significance_level
        self.pca = None
        self.T2_threshold = None
        self.Q_threshold = None
        self.virtual_tags = []

        self.fit_pca_model()
        self.construct_virtual_tags()

    def load_network(self, json_path):
        parser = JSONParser(json_path)
        self.network = parser.initialize_network()
        print(f'Loaded network from {json_path}')

    def pca_scree_plot(self, data):
        """Plot the scree plot to visualize the explained variance of each principal component."""
        pca = PCA(n_components=data.shape[1])
        pca.fit(data)
        explained_variance = pca.explained_variance_ratio_
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        plt.title('Scree Plot')
        plt.grid()
        plt.show()

    def fit_pca_model(self):
        """Fit the PCA model to the training data and calculate thresholds for T² and Q statistics."""
        print(f'Fitting PCA model with {self.n_components} components...')
        # Scale and fit PCA model
        # self.pca_scree_plot(self.training_data)
        self.pca = PCA(n_components=self.n_components)
        train_data = self.dataset.training_data_filtered.drop(columns=['timestamp'])
        self.scores = self.pca.fit_transform(train_data)

        # Calculate T² threshold
        eigenvalues = self.pca.explained_variance_
        a = self.n_components
        m = train_data.shape[0]
        self.T2_threshold = (a*(m-1)/(m-a)) * f.ppf(1 - self.significance_level, dfn=a, dfd=m - a)
        self.T2_threshold_1 = (a*(m-1)/(m-a)) * f.ppf(1 - 0.01, dfn=a, dfd=m - a)
        self.T2_threshold_5 = (a*(m-1)/(m-a)) * f.ppf(1 - 0.05, dfn=a, dfd=m - a)
        print(f'T² Threshold: {self.T2_threshold}')

        # Calculate Q threshold based on the residual sum of squares
        residuals = train_data - self.pca.inverse_transform(self.scores)
        residual_var = np.var(residuals, axis=0).sum()
        self.Q_threshold = residual_var * f.ppf(1 - self.significance_level, dfn=1, dfd=(m - a))
        self.Q_threshold_1 = residual_var * f.ppf(1 - 0.01, dfn=1, dfd=(m - a))
        self.Q_threshold_5 = residual_var * f.ppf(1 - 0.05, dfn=1, dfd=(m - a))
        print(f'Q Threshold: {self.Q_threshold}')

    def plot_pc(self, save=None):
        """
            Plot a biplot of the first two principal components.
            1. vector
            2. data points scaled by the variance
        """
        plt.figure(figsize=(8, 8))
        
        # biplot
        ax1 = plt.subplot(1, 1, 1)
        scaling_factor = 5
        ax1.scatter(self.scores[:, 0]/scaling_factor, self.scores[:, 1]/scaling_factor, c='yellow', alpha=0.3, s=2)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')

        # Plot variable vectors
        for i, (pc1, pc2) in enumerate(zip(self.pca.components_[0], self.pca.components_[1])):
            ax1.arrow(0, 0, pc1, pc2, color=RO_name_to_color[self.dataset.training_data.columns[i+1]], alpha=0.5)
            ax1.text(pc1, pc2, self.dataset.training_data.columns[i+1], color=RO_name_to_color[self.dataset.training_data.columns[i+1]], fontsize=12)
        ax1.set_title('PCA Biplot', fontsize=16)
        ax1.grid()

        plt.tight_layout()
        if save:
            plt.savefig(save)
        else:
            plt.show()

    def detect_faults(self, virtual_tags=False):
        """
        Detect faults by computing T² and Q statistics on new data.

        Parameters:
        - new_data: DataFrame containing new observations
        - virtual_tags: Whether to use VirtualTags to construct the PC domain

        Returns:
        - DataFrame with T² and Q statistics for each observation
        """
        # Project new data onto PCA model and calculate scores
        timestamp = self.dataset.testing_data_filtered['timestamp']
        new_data = self.dataset.testing_data_filtered.drop(columns=['timestamp'])
        if virtual_tags:
            scores_new = []
            for vt in self.virtual_tags:
                scores_new.append(vt.process_ops(new_data, RO_tag_mappping))
            scores_new = np.array(scores_new).T
            print(f'Projected new data onto PCA model using VirtualTags: {scores_new.shape}')

        else:   
            scores_new = self.pca.transform(new_data)

        T2_stats = np.sum((scores_new / np.sqrt(self.pca.explained_variance_))**2, axis=1)
        residuals = new_data - self.pca.inverse_transform(scores_new)
        Q_stats = np.sum(residuals**2, axis=1)

        T2_stats_full_length = np.zeros(self.dataset.testing_data.shape[0])
        Q_stats_full_length = np.zeros(self.dataset.testing_data.shape[0])

        # insert 0 in the index not in the test_index
        c = 0
        for i in self.dataset.testing_data.index:
            if i not in self.dataset.test_index:
                T2_stats_full_length[i] = T2_stats[c]
                Q_stats_full_length[i] = Q_stats[i]
                c += 1

        return pd.DataFrame({'T2': T2_stats_full_length, 'Q': Q_stats_full_length, 'T2_threshold': self.T2_threshold, 'Q_threshold': self.Q_threshold})
        # return pd.DataFrame({'T2': T2_stats, 'Q': Q_stats, 'T2_threshold': self.T2_threshold, 'Q_threshold': self.Q_threshold})

    def plot_fault_detection(self, stats_df, save=None):
        """Plot T² and Q statistics to visualize faults."""
        # reset index for plotting
        stats_df.reset_index(drop=True, inplace=True)
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={'hspace': 0.2})
        # Plot T² statistics
        axs[0].plot(stats_df['T2'])
        for i, (start, end) in enumerate(self.dataset.off_hours):
            axs[0].axvspan(start, end, color='gray', alpha=0.3, zorder=2)
        axs[0].set_ylim(0, 25)
        axs[0].axhline(y=self.T2_threshold_1, color='r', linestyle='--', label='α=0.01')
        axs[0].axhline(y=self.T2_threshold_5, color='orange', linestyle='--', label='α=0.05')
        axs[0].set_ylabel('T² Statistic')

        # Plot Q statistics
        axs[1].plot(stats_df['Q'])
        for i, (start, end) in enumerate(self.dataset.off_hours):
            axs[1].axvspan(start, end, color='gray', alpha=0.3, zorder=2)
        axs[1].set_ylim(0, 40)
        axs[1].axhline(y=self.Q_threshold_1, color='r', linestyle='--', label='α=0.01')
        axs[1].axhline(y=self.Q_threshold_5, color='orange', linestyle='--', label='α=0.05')
        axs[1].set_ylabel('Q Statistic')

        handles, labels = [], []
        for i in range(2): 
            h, l = axs[i].get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        label_to_handle = dict(zip(labels, handles))
        axs[0].legend(label_to_handle.values(), label_to_handle.keys(), bbox_to_anchor=(0.85, 1), loc='upper left')
        
        labels = self.dataset.testing_data['timestamp'].apply(lambda x: x[5:10])
        positions = range(0, len(labels)+1, len(labels)//10)
        plt.xticks(ticks=positions, labels=labels[::len(labels)//10])
        plt.xlabel('Timestamp')

        plt.suptitle('Fault Detection')
        plt.tight_layout()

        if save:
            plt.savefig(save)
        else:
            plt.show()

    def construct_virtual_tags(self):
        """
        Construct VirtualTag objects for the first 2 Principal Components (PCs) and T² and Q statistics.
        """
        # Get PCA coefficients (loadings) and sensor tags
        coefficients = self.pca.components_  # Shape: (num_pcs, num_features)
        tags = self.dataset.data_info['Tag'].tolist()  # List of sensor tags

        for pc_index in range(min(2, self.n_components)):
            # Get the coefficients for the current principal component
            pc_coeffs = coefficients[pc_index]
            
            # Construct the lambda operation string, use the varible name x_1, x_2, ... for each sensor tag
            operations = 'lambda ' + ', '.join("x_{} ".format(i+1) for i in range(len(tags))) + ': ' + ' + '.join(
                [f'({coef:.5f} * x_{i+1})' for i, (coef, tag) in enumerate(zip(pc_coeffs, tags)) if abs(coef) > 1e-5]
            )
            
            # Create a VirtualTag for the current principal component
            virtual_tag = VirtualTag(
                id=f'PC_{pc_index+1}',
                tags=[self.network.get_tag(tag) for tag in tags],
                operations=operations,
                tag_type='PCA_Component',
                parent_id='PC_Domain'
            )

            self.virtual_tags.append(virtual_tag)
            print(f'Created VirtualTag for PC_{pc_index+1}: {operations}')

        # Construct VirtualTag for T² statistic
        # scores_new = [PC_1, PC_2]
        # T2_stats = np.sum((scores_new / np.sqrt(self.pca.explained_variance_))**2, axis=1)
        # operations = 'lambda pc_1, pc_2: np.sum((np.array([pc_1, pc_2]) / np.sqrt({self.pca.explained_variance_}))**2, axis=1)'
        # virtual_tag = VirtualTag(
        #     id='T2',
        #     tags=['PC_1', 'PC_2'],
        #     operations=operations,
        #     tag_type='T2_Statistic',
        #     parent_id='T2Q_Domain'
        # )
        # self.virtual_tags.append(virtual_tag)
        # print(f'Created VirtualTag for T2: {operations}')

        # Construct VirtualTag for Q statistic
        # residuals = new_data - self.pca.inverse_transform(scores_new)
        # Q_stats = np.sum(residuals**2, axis=1)
        # operations = 'lambda ' + ', '.join("x_{} ".format(i+1) for i in range(len(tags))) + ': ' + ' + '.join(
        #     [f'({coef:.5f} * x_{i+1})' for i, (coef, tag) in enumerate(zip(pc_coeffs, tags)) if abs(coef) > 1e-5]
        # )
        # virtual_tag = VirtualTag(
        #     id='Q',
        #     tags=[self.network.get_tag(tag) for tag in tags],
        #     operations=operations,
        #     tag_type='Q_Statistic',
        #     parent_id='T2Q_Domain'
        # )
        # self.virtual_tags.append(virtual_tag)
        # print(f'Created VirtualTag for Q: {operations}')


if __name__ == '__main__':
    columns = ['timestamp', 
               'intake flowrate (GPM)',
               'intake conductivity (uS/cm)', 
               'wastewater flowrate (GPM)', 
               'HP Pump speed (RPM)', 
               'HP Pump pressure (PSI)', 
               'Circulation Pump speed (RPM)',
                'Circulation Pump pressure (PSI)'
               ]
    dataset = MeasurementData(tag_file=args.tag_file, 
                              data_folder=args.data_folder, 
                              train_test_split=args.train_test_split, 
                              columns=columns
                              )
    # dataset.print_data()
    # dataset.print_format()

    dataset.preprocess_data(remove_outliers=1, 
                            filter_data=1, 
                            # columns=columns
                            )
    
    # dataset.plot_data(save=args.data_plot)
    dataset.plot_data(save=args.data_plot.replace('.png', '_test.png'), mode='test')

    fd_system = FaultDetectionSystem('json/Desal.json',
                                     dataset, 
                                     n_components=2, 
                                     significance_level=0.01, 
                                     )
    stats_df = fd_system.detect_faults(virtual_tags=True)
    fd_system.plot_pc(save=args.PC_plot)
    fd_system.plot_fault_detection(stats_df, save=args.T2Q_plot)

    for vt in fd_system.virtual_tags:
        print(vt)
