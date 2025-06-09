from sklearn.metrics import r2_score, \
    mean_absolute_error, \
    median_absolute_error, \
    mean_squared_error, \
    mean_squared_log_error
from scipy import stats as st
from scipy.stats import t
import numpy as np
import pandas as pd
import scipy.stats as stats


class ConfidentIntervals:

    def _init__(self):
        pass

    def confidence_intervals_v1(self, true_data, model_data, scale=1.96):
        mae = mean_absolute_error(true_data, model_data)
        deviation = np.std(true_data - model_data)
        lower_bond = (mae + scale * deviation)
        upper_bond = (mae + scale * deviation)
        return lower_bond, upper_bond

    def confidence_intervals_v1_1(self, true_data, model_data, scale=1.96):
        mean = np.mean(abs(true_data - model_data))
        var = np.var(abs(true_data - model_data))
        deviation = np.std(true_data - model_data)
        # lower_bond = mean - 1.96*var/(len(true_data)*0.5)
        # upper_bound = mean + 1.96*var/(len(true_data)*0.5)
        lower_bond = mean - 1.96 * deviation
        upper_bound = mean + 1.96 * deviation
        return lower_bond, upper_bound

    def bootstrap_ci_2(self, true_data, model_data, size=10000, interval=95):
        abs_error = np.array(abs(true_data - model_data))
        bs_list = []
        for i in range(size):
            bs_list.append(np.mean(np.random.choice(abs_error, len(abs_error))))
        left_quantile = (100 - interval) / 2
        right_quantile = interval + (100 - interval) / 2
        lower_bond = np.percentile(bs_list, [left_quantile])
        upper_bound = np.percentile(bs_list, [right_quantile])
        return lower_bond, upper_bound

    def bootstrap_ci_3(self, true_data, model_data, size=1000, alpha=0.5):
        abs_error = np.array(abs(true_data - model_data))
        n = abs_error.size
        bootstrap_samples = np.random.choice(abs_error, (size, n), replace=True)
        theta_hat_star = np.mean(bootstrap_samples, axis=1)
        left, right = np.quantile(theta_hat_star, (alpha / 2, 1 - alpha / 2))

        lower_bond = left
        upper_bound = right

        return lower_bond, upper_bound

    def bootstrap_ci_hall(self, true_data, model_data, size=1000, interval=95):
        """
        Calculation Confident interval for model using Holt's bootstarp method
        :param true_data: real test data
        :param model_data: predicted data
        :param size: how many boostrap lists need to create
        :param interval: required probability. 95% by default
        :return: lower and upper bound of confident interavl
        """
        abs_error = np.array(abs(true_data - model_data))
        mae = np.mean(abs_error)
        bs_list = []
        for i in range(size):
            bs_list.append(np.mean(np.random.choice(abs_error, len(abs_error))))
        bs_list = bs_list - mae
        right_quantile = (100 - interval) / 2
        left_quantile = interval + (100 - interval) / 2
        lower_bond = mae - np.percentile(bs_list, [left_quantile])
        upper_bound = mae - np.percentile(bs_list, [right_quantile])
        return lower_bond, upper_bound

    def stats_ci(self, true_data, model_data, confidence=0.05):
        """
        Calculation Confident interval for model using Holt's bootstarp method
        :param true_data: real test data
        :param model_data: predicted data
        :param size: how many boostrap lists need to create
        :param interval: required probability. 95% by default
        :return: lower and upper bound of confident interavl
        """
        abs_error = np.array(abs(true_data - model_data))
        n = len(abs_error)
        mean = np.mean(abs_error)
        std_err = np.std(abs_error, ddof=1)
        # ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=std_err)
        ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=std_err / np.sqrt(n))
        lower_bond = ci[0]
        upper_bound = ci[1]
        return lower_bond, upper_bound

    def ci_for_classification(self, true_data, model_data):
        df = pd.DataFrame()
        df['Raw_Data'] = true_data
        df['Y_Predicted'] = model_data
        df['Error'] = df['Raw_Data'] - df['Y_Predicted']
        df['-3s'] = df['Error'].mean() - (2 * df['Error'].std())
        df['3s'] = df['Error'].mean() + (2 * df['Error'].std())
        df['-2s'] = df['Error'].mean() - (1.75 * df['Error'].std())
        df['2s'] = df['Error'].mean() + (1.75 * df['Error'].std())
        df['-1s'] = df['Error'].mean() - (1.5 * df['Error'].std())
        df['1s'] = df['Error'].mean() + (1.5 * df['Error'].std())
        print(df)

    def print_all(self, true_data, model_data):
        print(f"[confidence_intervals_v1] : {self.confidence_intervals_v1(true_data, model_data, scale=1.96)}")
        print(f"[confidence_intervals_v1_1] : {self.confidence_intervals_v1_1(true_data, model_data, scale=1.96)}")
        print(f"[Stats_CI] : {self.stats_ci(true_data, model_data)}")
        print(f"[BootStep_Hall] : {self.bootstrap_ci_hall(true_data, model_data, size=1000, interval=95)}")
        print(f"[bootstrap_ci_2] : {self.bootstrap_ci_2(true_data, model_data, size=10000, interval=95)}")
        print(f"[bootstrap_ci_3] : {self.bootstrap_ci_3(true_data, model_data, size=10000)}")
        print(f"[MAE=mean_absolute_error(true_data, model_data)] : {mean_absolute_error(true_data, model_data)}")
        print(f"[MEAN=np.mean(abs(true_data - model_data))]: {np.mean(abs(true_data - model_data))}")
        print(f"[deviation = np.std(true_data - model_data)]: {np.std(true_data - model_data)}")
        print(f"[var = np.var(abs(true_data - model_data))]: {np.var(abs(true_data - model_data))}")
