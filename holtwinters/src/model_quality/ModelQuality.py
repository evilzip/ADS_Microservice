from sklearn.metrics import median_absolute_error, \
    mean_squared_error, \
    mean_squared_log_error, \
    mean_absolute_percentage_error, \
    root_mean_squared_error, \
    r2_score, \
    median_absolute_error, \
    mean_absolute_error
import pandas as pd


class ModelQuality:

    def _init__(self):
        pass

    def model_quality_df(self, y_test, y_predicted):
        r2 = r2_score(y_test, y_predicted)
        mape = mean_absolute_percentage_error(y_test, y_predicted)
        rmse = root_mean_squared_error(y_test, y_predicted)
        mse = mean_squared_error(y_test, y_predicted)
        dict_data = {
            'MAPE': mape,
            'RMSE': rmse,
            'MSE': mse,
            'R2': r2
        }
        model_quality_df = pd.DataFrame([[mape, rmse, mse, r2]], columns=['MAPE', 'RMSE', 'MSE', 'R2'])
        return model_quality_df

