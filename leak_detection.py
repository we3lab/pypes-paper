from pype_schema.parse_json import JSONParser
from pype_schema.tag import VirtualTag
from define import RO_tag_mappping, RO_name_to_color

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import f
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--tag_file', type=str, default='data/SB_data/data_info.csv', help='Path to the tag file')
parser.add_argument('--data_folder', type=str, default='data/SB_data/case_study_data', help='Path to the data folder')
parser.add_argument('--train_test_split', type=float, default=0.7, help='Train-test split ratio')
args = parser.parse_args()

class MeasurementData:
    def __init__(self, tag_file, data_folder):
        self.data_folder = data_folder
        self.tag_file = tag_file
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
        data = pd.concat([pd.read_csv(os.path.join(self.data_folder, file)) for file in data_files])
        # remove "To date" column
        data.drop(columns=['To date'], inplace=True)
        # reset index and column names
        data.reset_index(drop=True, inplace=True)
        data.columns = [RO_tag_mappping[col] for col in data.columns]
        # remove nan
        data.dropna(how='all', inplace=True)
        print(f"Data loaded from {self.data_folder} with shape: {data.shape}")
        return data
    
    def print_data(self):
        '''
        Show the top 5 rows of the data
        '''
        print(self.data.head())

    def print_format(self):
        '''
        Show the data format from the tag file
        '''
        print(self.data_info)

    def plot_column(self, column):
        '''
        Plot the specified column data
        '''
        plt.figure(figsize=(10, 5))
        plt.plot(self.data[column], label=column)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Time Series Plot of {column}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def filter_data(self):
        # use intake flowrate > 500 to filter data
        self.data = self.data[self.data['intake flowrate'] > 500]

    def preprocess_data(self, columns=None, fill_missing='ffill', scale_data=True, remove_outliers=True, train_test_split=None, num=None):
        '''
        Preprocess the data with the following options:
        - fill_missing: Method to handle missing values ('ffill', 'bfill', or None)
        - scale_data: Standardize data to mean 0 and variance 1
        - remove_outliers: Remove outliers using the IQR method
        - columns: Specify which columns to preprocess (default is all numeric columns)
        - train_test_split: Split the data into training and test sets (e.g., 0.8 for 80% training data
        '''
        if num:
            self.data = self.data[:num]
        
        if columns:
            self.data = self.data[columns]
        else:
            # Select only numeric columns if not specified
            self.data = self.data.select_dtypes(include=[np.number])
        
        # Step 2: Handle missing values
        if fill_missing == 'ffill':
            self.data.fillna(method='ffill', inplace=True)
        elif fill_missing == 'bfill':
            self.data.fillna(method='bfill', inplace=True)
        elif fill_missing == 'mean':
            self.data.fillna(self.data.mean(), inplace=True)
        elif fill_missing == 'drop':
            self.data.dropna(inplace=True)

        if train_test_split:
            m = len(self.data)
            self.training_data = self.data[:int(m * train_test_split)]
            self.testing_data = self.data[int(m * train_test_split):]

        # Step 3: Remove outliers using the IQR method in train data
        if remove_outliers:
            Q1 = self.training_data.quantile(0.25)
            Q3 = self.training_data.quantile(0.75)
            IQR = Q3 - Q1
            self.training_data = self.training_data[~((self.training_data < (Q1 - 1.5 * IQR)) | (self.training_data > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # Step 4: Standardize the data (mean=0, variance=1)
        if scale_data:
            scaler = StandardScaler()
            self.training_data = pd.DataFrame(scaler.fit_transform(self.training_data), columns=self.training_data.columns)
            if self.testing_data is not None:
                self.testing_data = pd.DataFrame(scaler.transform(self.testing_data), columns=self.testing_data.columns)
            print("training_data shape: ", self.training_data.shape)
            print("testing_data shape: ", self.testing_data.shape)
        
        print("Preprocessing complete.")
        return self.data   

    def plot_type(self, data_type):
        '''
        Plot the data based on the data type
        '''
        data_columns = self.data_info[self.data_info['type'] == data_type]['Tag']
        data = self.data[data_columns]
        data.plot(subplots=True, figsize=(15, 10))
        plt.suptitle(f'Process Data for {data_type}')
        plt.show()  
    
    def plot_data(self):
        '''
        Plot the testing data in a vertical stack with same x axis
        '''
        plt.figure(figsize=(12, 8))
        axes = []
        plot_data = self.data.copy()
        plot_data = plot_data[RO_tag_mappping.keys()]
        num_of_columns = len(plot_data.columns)
        for i, col in enumerate(plot_data):
            ax = plt.subplot(num_of_columns, 1, i+1)
            ax.plot(plot_data[col], label=col)
            if i > 0:
                ax.sharex(axes[0])
            ax.legend()
            axes.append(ax)
        # plt.suptitle('Testing Data')
        plt.tight_layout()
        plt.show()

class FaultDetectionSystem:
    def __init__(self, json_path, dataset, n_components=2, significance_level=0.01):
        """
        Initialize with training data to build a PCA model.

        Parameters:
        - training_data: DataFrame containing training samples for normal operation
        - n_components: Number of principal components to retain
        - significance_level: Significance level for threshold calculation (e.g., 0.01 for 99% confidence)
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
        self.scores = self.pca.fit_transform(self.dataset.training_data)

        # Calculate T² threshold
        eigenvalues = self.pca.explained_variance_
        a = self.n_components
        m = self.dataset.training_data.shape[0]
        self.T2_threshold = (a*(m-1)/(m-a)) * f.ppf(1 - self.significance_level, dfn=a, dfd=m - a)
        print(f'T² Threshold: {self.T2_threshold}')

        # Calculate Q threshold based on the residual sum of squares
        residuals = self.dataset.training_data - self.pca.inverse_transform(self.scores)
        residual_var = np.var(residuals, axis=0).sum()
        self.Q_threshold = residual_var * f.ppf(1 - self.significance_level, dfn=1, dfd=(m - a))
        print(f'Q Threshold: {self.Q_threshold}')

    def plot_pc(self):
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
            ax1.arrow(0, 0, pc1, pc2, color=RO_name_to_color[self.dataset.training_data.columns[i]], alpha=0.5)
            ax1.text(pc1, pc2, self.dataset.training_data.columns[i], color=RO_name_to_color[self.dataset.training_data.columns[i]], fontsize=12)
        ax1.set_title('PCA Biplot')
        ax1.grid()

        plt.tight_layout()
        plt.show()

    def detect_faults(self):
        """
        Detect faults by computing T² and Q statistics on new data.

        Parameters:
        - new_data: DataFrame containing new observations

        Returns:
        - DataFrame with T² and Q statistics for each observation
        """
        # Project new data onto PCA model and calculate scores
        new_data = self.dataset.testing_data
        scores_new = self.pca.transform(new_data)
        T2_stats = np.sum((scores_new / np.sqrt(self.pca.explained_variance_))**2, axis=1)
        residuals = new_data - self.pca.inverse_transform(scores_new)
        Q_stats = np.sum(residuals**2, axis=1)

        return pd.DataFrame({'T2': T2_stats, 'Q': Q_stats, 'T2_threshold': self.T2_threshold, 'Q_threshold': self.Q_threshold})

    def plot_fault_detection(self, stats_df):
        """Plot T² and Q statistics to visualize faults."""
        plt.figure(figsize=(12, 5))

        # Plot T² statistics
        plt.subplot(1, 2, 1)
        plt.plot(stats_df['T2'], label='T²')
        plt.axhline(y=self.T2_threshold, color='r', linestyle='--', label='T² Threshold')
        plt.xlabel('Observation')
        plt.ylabel('T² Statistic')
        plt.legend()
        plt.title('T² Fault Detection')

        # Plot Q statistics
        plt.subplot(1, 2, 2)
        plt.plot(stats_df['Q'], label='Q')
        plt.axhline(y=self.Q_threshold, color='r', linestyle='--', label='Q Threshold')
        plt.xlabel('Observation')
        plt.ylabel('Q Statistic')
        plt.legend()
        plt.title('Q Fault Detection')

        plt.tight_layout()
        plt.show()

    def construct_virtual_tags(self):
        """
        Construct VirtualTag objects for the first 2 Principal Components (PCs).
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


if __name__ == '__main__':
    dataset = MeasurementData(tag_file=args.tag_file, data_folder=args.data_folder)
    # dataset.print_data()
    # dataset.print_format()
    dataset.filter_data()
    # dataset.plot_data()

    dataset.preprocess_data(fill_missing='mean', 
                            scale_data=True, 
                            remove_outliers=True, 
                            train_test_split=args.train_test_split, 
                            # num=50000
                            )

    fd_system = FaultDetectionSystem('json/Desal.json',
                                     dataset, 
                                     n_components=2, 
                                     significance_level=0.01)
    stats_df = fd_system.detect_faults()
    fd_system.plot_pc()
    fd_system.plot_fault_detection(stats_df)

    for vt in fd_system.virtual_tags:
        print(vt)
