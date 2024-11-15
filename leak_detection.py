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
parser.add_argument('--train_test_split', type=float, default=0.8, help='Train-test split ratio')
args = parser.parse_args()

class MeasurementData:
    def __init__(self, tag_file, data_folder):
        self.data_folder = data_folder
        self.tag_file = tag_file
        self.data_info = pd.read_csv(self.tag_file)
        self.data = self.load_data()
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        '''
        Load the data according to the tag file
        The tag file format:
        Tag, unit, type, node_id
        '''
        data_files = os.listdir(self.data_folder)
        # Load all CSV files in the data folder
        data = pd.concat([pd.read_csv(os.path.join(self.data_folder, file)) for file in data_files])
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

    def preprocess_data(self, columns=None, fill_missing='ffill', scale_data=True, remove_outliers=True, train_test_split=None):
        '''
        Preprocess the data with the following options:
        - fill_missing: Method to handle missing values ('ffill', 'bfill', or None)
        - scale_data: Standardize data to mean 0 and variance 1
        - remove_outliers: Remove outliers using the IQR method
        - columns: Specify which columns to preprocess (default is all numeric columns)
        - train_test_split: Split the data into training and test sets (e.g., 0.8 for 80% training data
        '''
        
        # Step 1: Filter the columns
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
            self.train_data = self.data[:int(m * train_test_split)]
            self.test_data = self.data[int(m * train_test_split):]

        # Step 3: Remove outliers using the IQR method in train data
        if remove_outliers:
            Q1 = self.train_data.quantile(0.25)
            Q3 = self.train_data.quantile(0.75)
            IQR = Q3 - Q1
            self.train_data = self.train_data[~((self.train_data < (Q1 - 1.5 * IQR)) | (self.train_data > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # Step 4: Standardize the data (mean=0, variance=1)
        if scale_data:
            scaler = StandardScaler()
            self.train_data = pd.DataFrame(scaler.fit_transform(self.train_data), columns=self.train_data.columns)
            if self.test_data is not None:
                self.test_data = pd.DataFrame(scaler.transform(self.test_data), columns=self.test_data.columns)
        
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
    
    def plot_testing_data(self):
        '''
        Plot the testing data in a vertical stack with same x axis
        '''
        plt.figure(figsize=(12, 8))
        num_of_columns = len(self.test_data.columns)
        axes = []
        for i, col in enumerate(self.test_data.columns):
            ax = plt.subplot(num_of_columns, 1, i+1)
            ax.plot(self.test_data[col], label=col)
            if i > 0:
                ax.sharex(axes[0])
            ax.legend()
            ax.set_ylabel(col)
            axes.append(ax)

        plt.suptitle('Testing Data')
        plt.show()



class FaultDetectionSystem:
    def __init__(self, training_data, n_components=3, significance_level=0.01):
        """
        Initialize with training data to build a PCA model.

        Parameters:
        - training_data: DataFrame containing training samples for normal operation
        - n_components: Number of principal components to retain
        - significance_level: Significance level for threshold calculation (e.g., 0.01 for 99% confidence)
        """
        self.training_data = training_data
        self.n_components = n_components
        self.significance_level = significance_level
        self.pca = None
        self.T2_threshold = None
        self.Q_threshold = None
        self.fit_pca_model()

    def fit_pca_model(self):
        """Fit the PCA model to the training data and calculate thresholds for T² and Q statistics."""
        print(f'Fitting PCA model with {self.n_components} components...')
        # Scale and fit PCA model
        self.pca = PCA(n_components=self.n_components)
        self.scores = self.pca.fit_transform(self.training_data)

        # Calculate T² threshold
        eigenvalues = self.pca.explained_variance_
        a = self.n_components
        m = self.training_data.shape[0]
        self.T2_threshold = (a*(m-1)/(m-a)) * f.ppf(1 - self.significance_level, dfn=a, dfd=m - a)
        print(f'T² Threshold: {self.T2_threshold}')

        # Calculate Q threshold based on the residual sum of squares
        residuals = self.training_data - self.pca.inverse_transform(self.scores)
        residual_var = np.var(residuals, axis=0).sum()
        self.Q_threshold = residual_var * f.ppf(1 - self.significance_level, dfn=1, dfd=(m - a))
        print(f'Q Threshold: {self.Q_threshold}')

    def detect_faults(self, new_data):
        """
        Detect faults by computing T² and Q statistics on new data.

        Parameters:
        - new_data: DataFrame containing new observations

        Returns:
        - DataFrame with T² and Q statistics for each observation
        """
        # Project new data onto PCA model and calculate scores
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

if __name__ == '__main__':
    dataset = MeasurementData(tag_file=args.tag_file, data_folder=args.data_folder)
    dataset.print_data()
    dataset.print_format()

    dataset.preprocess_data(fill_missing='mean', scale_data=True, remove_outliers=True, train_test_split=args.train_test_split)
    dataset.plot_testing_data()

    fd_system = FaultDetectionSystem(dataset.train_data, n_components=3, significance_level=0.01)
    stats_df = fd_system.detect_faults(dataset.test_data)
    fd_system.plot_fault_detection(stats_df)
