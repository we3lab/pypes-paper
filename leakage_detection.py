from pype_schema.parse_json import JSONParser
from pype_schema.tag import VirtualTag
from pype_schema.visualize import draw_graph

import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class LeakageDetectionSystem:
    def __init__(self, data_path, network_path, train_test_split=0.8):
        self.load_data(data_path, train_test_split)
        self.load_network(network_path)
        self.period = 365 * 24 * 12

    def load_data(self, path, train_test_split):
        '''
        Load data from a CSV file

        Timestamp   p227   p235  PUMP_1
        0  2018-01-01 00:00:00  77.77  83.93   44.04
        1  2018-01-01 00:05:00  72.51  76.34   44.06
        2  2018-01-01 00:10:00  71.54  78.68   44.06
        3  2018-01-01 00:15:00  69.85  75.50   44.07
        4  2018-01-01 00:20:00  77.12  82.99   44.03
        ...
        '''
        self.data = pd.read_csv(path)
        self.train_data = self.data.iloc[:int(train_test_split * len(self.data))]
        self.test_data = self.data.iloc[int(train_test_split * len(self.data)):]

        # initialize normalized data with same columns
        self.q_r = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        self.s_r = pd.DataFrame(index=self.data.index, columns=self.data.columns)
    
    def load_network(self, json_path):
        parser = JSONParser(json_path)
        self.network = parser.initialize_network()
    
    def seasonal_signal(self, node_name, terms=2, visualize=False, save_path=None):
        # if save_path file exists, load the coefficients
        if os.path.exists(save_path):
            self.q_r[node_name] = np.load(save_path)
            print(f'loaded seasonal history estimation from {save_path}')
            return
        
        omega = 2 * np.pi / self.period
        t = np.arange(len(self.data[node_name]))
        A = np.column_stack([np.ones_like(t)] + 
                            [np.cos(n * omega * t) for n in range(1, terms + 1)] + 
                            [np.sin(n * omega * t) for n in range(1, terms + 1)])
        
        def residuals(x):
            return A @ x - self.data[node_name]
        
        result = least_squares(residuals, np.zeros(2 * terms + 1))
        self.rho_coefficients = result.x

        estimated = A @ result.x

        # normalize along the x-axis
        self.q_r[node_name] = self.data[node_name]/estimated
        print(f'Mean of normalized data: {np.mean(self.q_r[node_name])}')

        if save_path:
            np.save(save_path, self.q_r[node_name])
        
        if visualize:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(self.data[node_name], label='q(k)')
            ax[0].plot(estimated, label='rho(k)')
            ax[0].legend()

            ax[1].plot(self.q_r[node_name], label='q_r(k)')
            ax[1].legend()
            plt.show()

    def calculate_seasonal_signal(self, nt, terms=2):
        omega = 2 * np.pi / self.period

        t = np.arange(nt)
        A = np.column_stack([np.ones_like(t)] + 
                            [np.cos(n * omega * t) for n in range(1, terms + 1)] + 
                            [np.sin(n * omega * t) for n in range(1, terms + 1)])
        return A @ self.rho_coefficients
    
    def test_seasonal_signal(self, node_name, visualize=False):
        data_gt = self.test_data[node_name]
        estimated = self.calculate_seasonal_signal(len(data_gt))
        mse = np.mean((data_gt - estimated) ** 2)
        print(f'Mean Squared Error: {mse}')
        print(estimated)
        # align the x-axis
        if visualize:
            plt.plot(data_gt, label='Ground Truth')
            plt.plot(estimated, label='Estimated')
            plt.legend()
            plt.show() 
    
    
    def weekly_signal(self, node_name, terms=100, visualize=False, save_path=None):
        if os.path.exists(save_path):
            self.theta0 = np.load(save_path)
            print(f'loaded weekly coffi from {save_path}')
            if visualize:
                first_wk = self.q_r[node_name][:2*7*24*12]
                estimated = self.calculate_weekly_signal(len(first_wk))
                plt.plot(first_wk, label='Ground Truth')
                plt.plot(estimated, label='Estimated')
                plt.xlabel('Time')
                plt.legend()
                plt.show()
            return
        omega = 2 * np.pi / self.period
        
        # use the mse opt solution from data in the first week to calculate the initial coefficients
        first_wk = self.q_r[node_name][:2*7*24*12]

        t = np.arange(len(first_wk))
        A = np.column_stack([np.ones_like(t)] +  # Constant term
                            [np.cos(n * omega * t) for n in range(1, terms + 1)] + 
                            [np.sin(n * omega * t) for n in range(1, terms + 1)])
        
       
        def residuals(x):
            return A @ x - first_wk
        
        self.theta0 = least_squares(residuals, np.zeros(2 * terms + 1)).x

        if save_path:
            np.save(save_path, self.theta0)

        if visualize:
            estimated = self.calculate_weekly_signal(len(first_wk))
            plt.plot(first_wk, label='Ground Truth')
            plt.plot(estimated, label='Estimated')
            plt.xlabel('Time')
            plt.legend()
            plt.show()
        
    def calculate_weekly_signal(self, nt, terms=100):
        omega = 2 * np.pi / self.period
        t = np.arange(nt)
        A = np.column_stack([np.ones_like(t)] + 
                            [np.cos(n * omega * t) for n in range(1, terms + 1)] + 
                            [np.sin(n * omega * t) for n in range(1, terms + 1)])
        return A @ self.theta0
    
    def update_theta(self, node_name, theta0, terms=100, G=None, alpha=0.01, visualize=False, learn=True):
        if G is None:
            G = 1e-3*np.eye(2*terms+1)

        omega = 2 * np.pi / self.period
        # 2 weeks
        data = self.q_r[node_name][:2*7*24*12]
        t = np.arange(len(data))
        zeta = np.column_stack([np.ones_like(t)] +  # Constant term
                               [np.cos(n * omega * t) for n in range(1, terms + 1)] + 
                               [np.sin(n * omega * t) for n in range(1, terms + 1)])
        
        if not learn:
            # repeat theta0 for all time steps in a lem(data) x 2*terms+1 matrix
            self.theta_r = np.zeros((len(data), 2*terms+1))
            for i in range(len(data)):
                self.theta_r[i] = theta0
            for i in range(len(data)):
                self.s_r[node_name][i] = np.dot(self.theta_r[i], zeta[i])
            if visualize:
                plt.plot(data, label='Ground Truth')
                plt.plot(self.s_r[node_name], label='Estimated')
                plt.xlabel('Time')
                plt.legend()
                plt.show()
            return

        theta_r = np.zeros((len(data), 2*terms+1))
        theta_r[0] = theta0
        print(f'Updating theta for {node_name}')
        for k in range(len(data)-1):
            zeta_k = zeta[k]
            s_r = np.dot(theta_r[k], zeta_k)
            e_r = data[k] - s_r
            theta_r[k+1] = theta_r[k] + G @ (zeta_k / (alpha + np.dot(zeta_k, zeta_k))) * e_r
            if k % 1000 == 0:
                print(f'k={k}, e_r={np.mean(e_r)}')
        self.theta_r = theta_r
        for i in range(len(data)):
            self.s_r[node_name][i] = np.dot(self.theta_r[i], zeta[i])

        if visualize:
            plt.plot(data, label='Ground Truth')
            plt.plot(self.s_r[node_name], label='Estimated')
            plt.legend()
            plt.show()

    def visualize_model(self, node_name):
        seasonal_gt = self.train_data[node_name]
        seasonal_estimated = self.calculate_seasonal_signal(len(seasonal_gt))
        weekly_gt = self.train_data.iloc[:, 1]
        weekly_estimated = self.calculate_weekly_signal(len(weekly_gt))

        estimated = seasonal_estimated + weekly_estimated
        mse = np.mean((self.train_data[node_name] - estimated) ** 2)
        print(f'Mean Squared Error: {mse}')
        plt.plot(self.train_data[node_name], label='Ground Truth')
        plt.plot(estimated, label='Estimated')
        plt.legend()
        plt.show() 

    def estimate_leakage_threshold(self, node_name):
        ''' Estimate the small leakage threshold (eta) from historical data. '''
        residuals = self.q_r[node_name] - 1  # Deviation from expected inflow
        eta = np.mean(residuals[residuals > 0])  # Consider only positive deviations
        print(f"Estimated eta (small leakage threshold): {eta}")
        return eta
    
    def detect_leakage(self, node_name, eta=None, threshold=30, weeks=2):
        ''' Implement CUSUM-based leakage detection. '''
        if eta is None:
            eta = self.estimate_leakage_threshold(node_name)
        
        # 1st term of the Fourier series
        theta0 = self.theta_r[:, 0]
        if weeks > 0:
            theta0 = theta0[:weeks*7*24*12]
        cusum = np.zeros_like(theta0)
        for k in range(1, len(theta0)):
            cusum[k] = max(0, cusum[k-1] + (theta0[k] - 1 - eta))
        
        detection_times = np.where(cusum > threshold)[0]
        print(f"Leak detected at indices: {detection_times}")
        return detection_times, cusum
    
    def visualize_leakage(self, node_name, weeks=2, threshold=30, save_path=None):
        ''' Visualize detected leakage events. '''
        detection_times, cusum = self.detect_leakage(node_name, weeks=weeks, threshold=threshold)
        normalized_inflow = self.q_r[node_name][:weeks*7*24*12]
        fig = plt.figure(figsize=(12, 6))
        plt.plot(normalized_inflow, label="Normalized Inflow")
        plt.plot(cusum / max(cusum), linestyle='dashed', label="CUSUM (scaled)")
        # for dt in detection_times:
        #     plt.axvline(dt, color='r', linestyle='dotted', label='Leak Detected' if dt == detection_times[0] else "")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Normalized Flow & CUSUM")
        plt.title(f"Leakage Detection for {node_name}")
        plt.show()
        if save_path:
            fig.savefig(save_path)

if __name__ == '__main__':
    node = 'p227'
    data_path = 'data/BattLeDIM/2018_SCADA_Flows.csv'
    network_path = 'json/L-Town.json'
    lds = LeakageDetectionSystem(data_path, network_path)
    
    lds.seasonal_signal(node_name=node, 
                        visualize=False, 
                        save_path=f'results/leak_detection/qr_{node}.npy')
    lds.weekly_signal(node_name=node, 
                      visualize=False, 
                      save_path=f'results/leak_detection/theta0_{node}.npy')
    lds.update_theta(node_name=node,
                     theta0=lds.theta0,
                     terms=100,
                     G=None,
                     alpha=0.01, 
                     visualize=False, 
                     learn=True)
    lds.visualize_leakage(node_name=node, 
                          threshold=50,
                          save_path=f'results/leak_detection/leakage_{node}.png')
    # lds.visualize_model(node_name='p227')
    # lds.test_seasonal_signal(node_name='p227', visualize=True)