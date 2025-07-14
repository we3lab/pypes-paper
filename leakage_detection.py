from pype_schema.parse_json import JSONParser
from pype_schema.tag import VirtualTag
from pype_schema.visualize import draw_graph

import os
import time
import numpy as np
from datetime import datetime
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator
import pandas as pd
from scipy.optimize import least_squares
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# Define the locator and formatter
plt.rcParams["figure.dpi"] = 300
xtick_locator = MonthLocator(interval=3)  # Show ticks every month
xtick_formatter = DateFormatter("%b %d")  # Format as "Feb 10"

node_colors = {
    "p227": "black",
    "p235": "#B2E08A",
}
plt.rcParams["axes.labelsize"] = 14  # Font size for axis labels
plt.rcParams["xtick.labelsize"] = 14  # Font size for x-axis tick labels
plt.rcParams["ytick.labelsize"] = 14  # Font size for y-axis tick labels


class LeakageDetectionSystem:
    def __init__(self, train_data_path, test_data_path, network_path, delta_t=5):
        self.load_data(train_data_path, test_data_path)
        self.load_network(network_path)
        self.Tr = (365 * 24 * 12) / delta_t
        self.Ts = (7 * 24 * 12) / delta_t
        self.timestamp = self.train_data.index

    def load_data(self, train_path, test_path):
        """
        Load data from a CSV file

        Timestamp   p227   p235  PUMP_1
        2018-01-01 00:00:00  77.77  83.93   44.04
        2018-01-01 00:05:00  72.51  76.34   44.06
        2018-01-01 00:10:00  71.54  78.68   44.06
        2018-01-01 00:15:00  69.85  75.50   44.07
        2018-01-01 00:20:00  77.12  82.99   44.03
        ...
        """
        self.train_data = pd.read_csv(train_path, index_col=0, parse_dates=True)
        self.test_data = pd.read_csv(test_path, index_col=0, parse_dates=True)

        # initialize normalized data with same columns
        self.q_r = pd.DataFrame(
            index=self.train_data.index, columns=self.train_data.columns
        )
        self.s_r = pd.DataFrame(
            index=self.train_data.index, columns=self.train_data.columns
        )
        self.test_q_r = pd.DataFrame(
            index=self.test_data.index, columns=self.test_data.columns
        )
        self.test_s_r = {c: None for c in self.test_data.columns}
        self.test_theta_r = {c: None for c in self.test_data.columns}

    def load_network(self, json_path):
        parser = JSONParser(json_path)
        self.network = parser.initialize_network()

    def seasonal_signal(
        self, node_name, terms=2, visualize=False, save_path=None, override=False
    ):
        omega = 2 * np.pi / self.Tr
        data = self.train_data[node_name]
        t = np.arange(len(data))
        A = np.column_stack(
            [np.ones_like(t)]
            + [np.cos(n * omega * t) for n in range(1, terms + 1)]
            + [np.sin(n * omega * t) for n in range(1, terms + 1)]
        )

        def residuals(x):
            return A @ x - data

        qr_path = f"{save_path}/qr_{node_name}.npy"
        rho_path = f"{save_path}/rho_{node_name}.npy"
        img_path = f"{save_path}/seasonal_{node_name}.png"

        # if save_path file exists, load the coefficients
        if os.path.exists(qr_path) and os.path.exists(rho_path) and not override:
            self.q_r[node_name] = np.load(qr_path)
            self.rho = np.load(rho_path)
            estimated = A @ self.rho
            print(f"loaded seasonal history estimation from {save_path}")
        else:
            print(f"Estimating seasonal history for {node_name}, length={len(data)}")
            result = least_squares(residuals, np.zeros(2 * terms + 1))
            self.rho = result.x
            estimated = A @ self.rho
            self.q_r[node_name] = data / estimated
            np.save(qr_path, self.q_r[node_name])
            np.save(rho_path, self.rho)

        if visualize:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            ax[0].xaxis.set_major_locator(xtick_locator)
            ax[0].xaxis.set_major_formatter(xtick_formatter)
            ax[0].plot(self.timestamp, data, label="q(k)")
            ax[0].plot(self.timestamp, estimated, label="rho(k)")
            ax[0].legend()

            ax[1].xaxis.set_major_locator(xtick_locator)
            ax[1].xaxis.set_major_formatter(xtick_formatter)
            ax[1].plot(self.q_r[node_name], label="q_r(k)")
            ax[1].legend()
            fig.suptitle(f"Seasonal history estimation for {node_name}")
            plt.show()
            fig.savefig(img_path, bbox_inches="tight")
            plt.close()

    def calculate_seasonal_signal(self, nt, terms=2):
        omega = 2 * np.pi / self.Tr

        t = np.arange(nt)
        A = np.column_stack(
            [np.ones_like(t)]
            + [np.cos(n * omega * t) for n in range(1, terms + 1)]
            + [np.sin(n * omega * t) for n in range(1, terms + 1)]
        )
        return A @ self.rho

    def test_seasonal_signal(self, node_name, visualize=False):
        data_gt = self.test_data[node_name]
        estimated = self.calculate_seasonal_signal(len(data_gt))
        mse = np.mean((data_gt - estimated) ** 2)
        print(f"Mean Squared Error: {mse}")
        print(estimated)
        # align the x-axis
        if visualize:
            plt.plot(data_gt, label="Ground Truth")
            plt.plot(estimated, label="Estimated")
            plt.legend()
            plt.show()

    def weekly_signal(
        self, node_name, terms=100, visualize=False, save_path=None, override=False
    ):
        omega = 2 * np.pi / self.Ts

        data = self.q_r[node_name]

        def residuals(x):
            return A @ x - data

        t = np.arange(len(data))
        A = np.column_stack(
            [np.ones_like(t)]  # Constant term
            + [np.cos(n * omega * t) for n in range(1, terms + 1)]
            + [np.sin(n * omega * t) for n in range(1, terms + 1)]
        )

        theta0_path = f"{save_path}/theta_{node_name}.npy"
        img_path = f"{save_path}/weekly_{node_name}.png"

        if os.path.exists(theta0_path) and not override:
            self.theta0 = np.load(theta0_path)
            print(f"loaded weekly estimation from {theta0_path}")

        else:
            print(f"Estimating weekly history for {node_name}, length={len(data)}")
            result = least_squares(residuals, np.zeros(2 * terms + 1))
            self.theta0 = result.x
            np.save(theta0_path, self.theta0)

        # visualize the first 2 weeks
        if visualize:
            estimated = A @ self.theta0
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))

            ax[0].xaxis.set_major_locator(xtick_locator)
            ax[0].xaxis.set_major_formatter(xtick_formatter)
            ax[0].plot(self.timestamp, data, label="q_r(k)")
            ax[0].plot(self.timestamp, estimated, label="s_r(k)")
            ax[0].legend()

            ax[1].xaxis.set_major_locator(xtick_locator)
            ax[1].xaxis.set_major_formatter(xtick_formatter)
            ax[1].plot(
                self.timestamp[: 2 * 7 * 24 * 12],
                data[: 2 * 7 * 24 * 12],
                label="q_r(k)",
            )
            ax[1].plot(
                self.timestamp[: 2 * 7 * 24 * 12],
                estimated[: 2 * 7 * 24 * 12],
                label="s_r(k)",
            )
            ax[1].legend()
            fig.suptitle(f"Weekly history estimation for {node_name}")
            plt.show()
            fig.savefig(img_path, bbox_inches="tight")
            plt.close()

    def calculate_weekly_signal(self, nt, terms=100):
        omega = 2 * np.pi / self.Ts
        t = np.arange(nt)
        A = np.column_stack(
            [np.ones_like(t)]
            + [np.cos(n * omega * t) for n in range(1, terms + 1)]
            + [np.sin(n * omega * t) for n in range(1, terms + 1)]
        )
        return A @ self.theta0

    def processing_testing_data(
        self,
        node_name,
        visualize=False,
        save_path=None,
        start_date=None,
        weeks=2,
        override=False,
    ):
        if start_date is not None:
            start_idx = (start_date - self.test_data.index[0]).days * 24 * 12
        else:
            start_idx = 0

        data = self.test_data[node_name]
        timestamp = self.test_data.index[start_idx : start_idx + weeks * 7 * 24 * 12]

        theta_r_path = f"{save_path}/theta_r_{node_name}.npy"
        s_r_path = f"{save_path}/s_r_{node_name}.npy"
        s_r_theta0_path = f"{save_path}/s_r_theta0_{node_name}.npy"
        q_r_path = f"{save_path}/q_r_{node_name}.npy"
        img_path = f"{save_path}/weekly_test_{node_name}.png"

        if (
            os.path.exists(theta_r_path)
            and os.path.exists(s_r_path)
            and os.path.exists(q_r_path)
            and os.path.exists(s_r_theta0_path)
            and not override
        ):
            self.test_theta_r[node_name] = np.load(theta_r_path)
            self.test_s_r[node_name] = np.load(s_r_path)
            s_r_theta0 = np.load(s_r_theta0_path)
            self.test_q_r[node_name] = np.load(q_r_path)
            print(f"loaded testing data from {save_path}")

        else:
            # update theta
            self.test_q_r[node_name] = data / self.calculate_seasonal_signal(len(data))
            self.test_theta_r[node_name], self.test_s_r[node_name], s_r_theta0 = (
                self.update_theta(
                    self.test_q_r[node_name],
                    terms=100,
                    G=None,
                    alpha=0.01,
                    start_idx=start_idx,
                    weeks=weeks,
                )
            )
            np.save(theta_r_path, self.test_theta_r[node_name])
            np.save(s_r_path, self.test_s_r[node_name])
            np.save(s_r_theta0_path, s_r_theta0)
            np.save(q_r_path, self.test_q_r[node_name])

        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.xaxis.set_major_locator(xtick_locator)
            ax.xaxis.set_major_formatter(xtick_formatter)
            ax.plot(
                timestamp,
                self.test_q_r[node_name][start_idx : start_idx + weeks * 7 * 24 * 12],
                label="q_r(k)",
            )
            ax.plot(timestamp, self.test_s_r[node_name], label="s_r(k)")
            ax.plot(timestamp, s_r_theta0, label="s_r_theta0(k)")
            ax.legend()
            fig.suptitle(f"Testing data estimation for {node_name}")
            plt.show()
            fig.savefig(img_path, bbox_inches="tight")
            plt.close()

    def update_theta(
        self,
        q_r,
        terms=100,
        G=None,
        alpha=0.01,
        start_idx=None,
        weeks=2,
        visualize=False,
    ):
        if G is None:
            G = 1e-2 * np.eye(2 * terms + 1)

        if start_idx is not None:
            q_r = q_r[start_idx : start_idx + weeks * 7 * 24 * 12]

        omega = 2 * np.pi / self.Ts
        s_r = np.zeros_like(q_r)
        t = np.arange(len(q_r))
        A = np.column_stack(
            [np.ones_like(t)]  # Constant term
            + [np.cos(n * omega * t) for n in range(1, terms + 1)]
            + [np.sin(n * omega * t) for n in range(1, terms + 1)]
        )

        s_r_theta0 = np.zeros_like(q_r)
        for i in range(len(q_r)):
            s_r_theta0[i] = A[i] @ self.theta0

        theta_r = np.zeros((len(q_r), 2 * terms + 1))
        theta_r[0] = self.theta0

        for k in range(len(q_r) - 1):
            s_r[k] = A[k] @ theta_r[k]
            e_r = q_r[k] - s_r[k]
            theta_r[k + 1] = theta_r[k] + G @ (A[k] / (alpha + A[k].T @ A[k])) * e_r
            if k % 10000 == 0:
                print(f"k={k}, e_r={np.mean(e_r)}")

        return theta_r, s_r, s_r_theta0

    def visualize_model(self, node_name):
        seasonal_gt = self.train_data[node_name]
        seasonal_estimated = self.calculate_seasonal_signal(len(seasonal_gt))
        weekly_gt = self.train_data.iloc[:, 1]
        weekly_estimated = self.calculate_weekly_signal(len(weekly_gt))

        estimated = seasonal_estimated + weekly_estimated
        mse = np.mean((self.train_data[node_name] - estimated) ** 2)
        print(f"Mean Squared Error: {mse}")
        plt.plot(self.train_data[node_name], label="Ground Truth")
        plt.plot(estimated, label="Estimated")
        plt.legend()
        plt.show()

    def estimate_leakage_threshold(
        self, node_name, ratio=1e-1, start_idx=None, weeks=None
    ):
        """Estimate the small leakage threshold (eta) from historical data."""
        residuals = (
            self.test_q_r[node_name][start_idx : start_idx + weeks * 7 * 24 * 12]
            - self.test_s_r[node_name]
        )
        if weeks is not None:
            residuals = residuals
        eta = ratio * np.mean(
            residuals[residuals > 0]
        )  # Consider only positive deviations
        print(f"Estimated eta (small leakage threshold): {eta}")
        return eta

    def detect_leakage(
        self, node_name, eta=None, threshold=30, start_idx=None, weeks=None
    ):
        """Implement CUSUM-based leakage detection."""
        if eta is None:
            eta = self.estimate_leakage_threshold(
                node_name, start_idx=start_idx, weeks=weeks
            )

        # 1st term of the Fourier series
        theta_0 = self.test_theta_r[node_name][:, 0]
        cusum = np.zeros_like(theta_0)
        for k in range(1, len(theta_0)):
            cusum[k] = max(0, cusum[k - 1] + (theta_0[k] - 1 - eta))

        detection_times = np.where(cusum > threshold)[0]
        print(f"Leak detected at indices: {detection_times}")
        return detection_times, cusum

    def visualize_leakage(
        self,
        node_name,
        start_date=None,
        weeks=2,
        threshold=1,
        save_path=None,
        visualize=False,
    ):
        """Visualize detected leakage events."""
        if start_date is not None:
            start_idx = start_idx = (
                (start_date - self.test_data.index[0]).days * 24 * 12
            )
        else:
            start_idx = 0

        detection_times, cusum = self.detect_leakage(
            node_name, start_idx=start_idx, weeks=weeks, threshold=threshold
        )
        normalized_inflow = self.test_q_r[node_name][: weeks * 7 * 24 * 12]
        timestamp = self.test_data.index
        if weeks is not None:
            timestamp = timestamp[: weeks * 7 * 24 * 12]

        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.xaxis.set_major_locator(DayLocator(interval=10))
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))

        ax.plot(
            timestamp,
            normalized_inflow,
            label=f"{node_name})",
            color=node_colors[node_name],
            linewidth=1.5,
        )
        ax.plot(
            timestamp,
            cusum / max(cusum),
            linestyle="dashed",
            label="CUSUM",
            color="#A6CEE4",
            linewidth=2,
        )
        ax.axhline(
            threshold / max(cusum),
            linestyle="dashed",
            label="Threshold",
            color="#fb9a99",
            linewidth=2,
        )
        ax.set_xticks(
            [
                datetime(2019, 1, 1),
                datetime(2019, 1, 15),
                datetime(2019, 1, 29),
                datetime(2019, 2, 12),
                datetime(2019, 2, 26),
                datetime(2019, 3, 12),
            ]
        )
        ax.set_ylabel("Normalized values")
        ax.set_ylim([-0.1, 2.0])
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.tick_params(axis="x", labelrotation=45)
        if visualize:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    nodes = ["p235", "p227"]  # p227, p235, PUMP_1
    weeks = 10
    train_path = "data/distribution/2018_SCADA_Flows.csv"
    test_path = "data/distribution/2019_SCADA_Flows.csv"
    network_path = "json/distribution.json"
    lds = LeakageDetectionSystem(train_path, test_path, network_path)
    for node in nodes:
        lds.seasonal_signal(
            node_name=node,
            terms=2,
            visualize=False,
            save_path=f"results/leak_detection",
            override=False,
        )
        lds.weekly_signal(
            node_name=node,
            terms=100,
            visualize=False,
            save_path=f"results/leak_detection",
            override=False,
        )

        lds.processing_testing_data(
            node_name=node,
            visualize=False,
            save_path=f"results/leak_detection",
            start_date=datetime(2019, 1, 1),
            weeks=weeks,
            override=False,
        )

        lds.visualize_leakage(
            node_name=node,
            threshold=5,
            start_date=datetime(2019, 1, 1),
            weeks=weeks,
            save_path=f"results/leak_detection/leakage_{node}.png",
            visualize=False,
        )
