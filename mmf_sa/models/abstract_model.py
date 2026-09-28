from abc import abstractmethod
import logging
import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
import mlflow
from mmf_sa.exceptions import UnsupportedMetricError

_logger = logging.getLogger(__name__)
mlflow.set_registry_uri("databricks-uc")

MMF_PACKAGE = "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git"

MODEL_PIP_REQUIREMENTS = {
    "neuralforecast": [
        "cloudpickle==2.2.1",
        "neuralforecast==3.1.4",
        "ray[tune]==2.5.0",
        MMF_PACKAGE,
    ],
    "chronos": [
        "torch>=2.3.1",
        "transformers>=4.41.2",
        "chronos-forecasting==2.2.2",
        MMF_PACKAGE,
    ],
    "timesfm": [
        "timesfm[torch,xreg] @ git+https://github.com/google-research/timesfm.git@a83dbf3f163cf15993dac6a45bbb5dcb160e14e8",
        MMF_PACKAGE,
    ],
    "moirai": [
        "uni2ts==2.0.0",
        MMF_PACKAGE,
    ],
}


class ForecastingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params):
        self.params = params
        self.freq = params["freq"].upper()[0]
        self.one_ts_offset = (
            pd.offsets.MonthEnd(1) if self.freq == "M" else
            pd.DateOffset(weeks=1) if self.freq == "W" else
            pd.DateOffset(days=1) if self.freq == "D" else
            pd.DateOffset(hours=1) if self.freq == "H" else
            None
        )
        self.prediction_length_offset = (
            pd.offsets.MonthEnd(params["prediction_length"]) if self.freq == "M" else
            pd.DateOffset(weeks=params["prediction_length"]) if self.freq == "W" else
            pd.DateOffset(days=params["prediction_length"]) if self.freq == "D" else
            pd.DateOffset(hours=params["prediction_length"]) if self.freq == "H" else
            None
        )

    # ── Prediction interval helpers ──────────────────────────────────────

    def _get_interval_level(self) -> Optional[float]:
        """Return the prediction interval level (e.g. 0.95) or None if disabled."""
        level = self.params.get("prediction_interval_level", None)
        if level is not None and 0 < float(level) < 1:
            return float(level)
        return None

    def _get_interval_level_pct(self) -> Optional[int]:
        """Return the interval level as an integer percentage (e.g. 95) for
        libraries that use that convention (StatsForecast)."""
        level = self._get_interval_level()
        return int(round(level * 100)) if level is not None else None

    @staticmethod
    def compute_conformal_intervals(
        all_residuals: np.ndarray,
        forecast: np.ndarray,
        level: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute conformal prediction intervals from accumulated backtest residuals.

        Uses the distribution-free split-conformal method: the interval half-width
        is the (1 - alpha) quantile of the absolute residuals collected from
        *previous* backtest windows.

        Args:
            all_residuals: 1-D array of absolute residuals from prior windows.
            forecast: 1-D array of point forecasts for the current window.
            level: Confidence level (e.g. 0.95).

        Returns:
            (lower, upper) arrays the same length as *forecast*.
        """
        if len(all_residuals) == 0:
            # First window — no prior residuals; return NaNs
            nans = np.full_like(forecast, np.nan, dtype=float)
            return nans, nans
        q = np.quantile(np.abs(all_residuals), level)
        return forecast - q, forecast + q

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def fit(self, x, y=None):
        pass

    @abstractmethod
    def predict(self, x, y=None):
        pass

    @abstractmethod
    def forecast(self, x, spark=None):
        pass

    def backtest(
            self,
            df: pd.DataFrame,
            start: pd.Timestamp,
            group_id: Union[str, int] = None,
            # backtest_retrain: bool = False,
            spark=None,
    ) -> pd.DataFrame:
        """
        Performs backtesting using the provided pandas DataFrame, start timestamp, group id, stride and SparkSession.
        When prediction_interval_level is set, also produces forecast_lower and forecast_upper.
        Native intervals are used when the model provides them; otherwise conformal prediction
        intervals are computed from the accumulated backtest residuals.
        """
        stride = int(self.params["stride"]) # Read in stride
        stride_offset = (
            pd.offsets.MonthEnd(stride) if self.freq == "M" else
            pd.DateOffset(weeks=stride) if self.freq == "W" else
            pd.DateOffset(days=stride) if self.freq == "D" else
            pd.DateOffset(hours=stride) if self.freq == "H" else
            None
        )
        df = df.copy().sort_values(by=[self.params["date_col"]])
        end_date = df[self.params["date_col"]].max() # Last date from the training data
        # Offsets the timestamp: e.g. if it's in the middle of the month for a monthly time series, makes it the end of the month
        curr_date = start + self.one_ts_offset

        interval_level = self._get_interval_level()
        accumulated_residuals: List[float] = []  # for conformal fallback
        results = []

        while curr_date + self.prediction_length_offset <= end_date + self.one_ts_offset:
            _df = df[df[self.params["date_col"]] < np.datetime64(curr_date)]
            actuals_df = df[
                (df[self.params["date_col"]] >= np.datetime64(curr_date))
                & (
                        df[self.params["date_col"]]
                        < np.datetime64(curr_date + self.prediction_length_offset)
                )]

            metrics = self.calculate_metrics(_df, actuals_df, curr_date, spark)

            if isinstance(metrics, dict):
                forecast_arr = metrics["forecast"]
                actual_arr = metrics["actual"]

                # Determine interval bounds
                forecast_lower = metrics.get("forecast_lower", None)
                forecast_upper = metrics.get("forecast_upper", None)

                if interval_level is not None:
                    if forecast_lower is None or forecast_upper is None:
                        # No native intervals — use conformal prediction
                        forecast_lower, forecast_upper = self.compute_conformal_intervals(
                            np.array(accumulated_residuals), forecast_arr, interval_level
                        )
                    # Accumulate residuals for future conformal windows
                    accumulated_residuals.extend(
                        np.abs(actual_arr.astype(float) - forecast_arr.astype(float)).tolist()
                    )

                evaluation_results = [
                    (
                        group_id,
                        metrics["curr_date"],
                        metrics["metric_name"],
                        metrics["metric_value"],
                        forecast_arr,
                        actual_arr,
                        metrics["model_pickle"],
                        forecast_lower if forecast_lower is not None else np.array([]),
                        forecast_upper if forecast_upper is not None else np.array([]),
                    )
                ]
                results.extend(evaluation_results)
            elif isinstance(metrics, list):
                # Global / NeuralForecast models return a list of tuples
                for m in metrics:
                    if interval_level is not None and len(m) == 7:
                        # Tuple without intervals — add conformal
                        _, _, _, _, f_arr, a_arr, _ = m
                        f_lower, f_upper = self.compute_conformal_intervals(
                            np.array(accumulated_residuals),
                            np.asarray(f_arr, dtype=float),
                            interval_level,
                        )
                        accumulated_residuals.extend(
                            np.abs(np.asarray(a_arr, dtype=float) - np.asarray(f_arr, dtype=float)).tolist()
                        )
                        results.append(m + (f_lower, f_upper))
                    elif len(m) >= 9:
                        # Already has intervals
                        results.append(m)
                    else:
                        # No intervals requested — pass through with empty arrays
                        results.append(m + (np.array([]), np.array([])))

            curr_date += stride_offset

        res_df = pd.DataFrame(
            results,
            columns=[self.params["group_id"],
                     "backtest_window_start_date",
                     "metric_name",
                     "metric_value",
                     "forecast",
                     "actual",
                     "model_pickle",
                     "forecast_lower",
                     "forecast_upper"],
        )

        return res_df

    def calculate_metrics(
            self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> Dict[str, Union[str, float, bytes]]:
        """
        Calculates the metrics using the provided historical DataFrame, validation DataFrame, current date, and SparkSession.
        Parameters:
            self (Forecaster): A Forecaster object.
            hist_df (pd.DataFrame): A pandas DataFrame.
            val_df (pd.DataFrame): A pandas DataFrame.
            curr_date: A pandas Timestamp object.
            spark (SparkSession, optional): A SparkSession object. Default is None.
        Returns: metrics (Dict[str, Union[str, float, bytes]]): A dictionary specifying the metrics.
        """
        pred_df, model_fitted = self.predict(hist_df, val_df)
        
        actual = val_df[self.params["target"]].to_numpy()
        forecast = pred_df[self.params["target"]].to_numpy()

        # Warn if MAPE/sMAPE is used with data that contains negative or zero values
        if self.params.get("allow_negative_values", False) and self.params["metric"] in ("mape", "smape"):
            if np.any(actual == 0) or np.any(actual < 0):
                _logger.warning(
                    f"Metric '{self.params['metric']}' may produce unreliable results with "
                    f"negative or zero actual values. Consider using 'mae', 'mse', or 'rmse' instead."
                )

        if self.params["metric"] == "smape":
            smape = MeanAbsolutePercentageError(symmetric=True)
            metric_value = smape(actual, forecast)
        elif self.params["metric"] == "mape":
            mape = MeanAbsolutePercentageError(symmetric=False)
            metric_value = mape(actual, forecast)
        elif self.params["metric"] == "mae":
            mae = MeanAbsoluteError()
            metric_value = mae(actual, forecast)
        elif self.params["metric"] == "mse":
            mse = MeanSquaredError(square_root=False)
            metric_value = mse(actual, forecast)
        elif self.params["metric"] == "rmse":
            rmse = MeanSquaredError(square_root=True)
            metric_value = rmse(actual, forecast)
        else:
            raise UnsupportedMetricError(f"Metric {self.params['metric']} not supported!")

        result = {
            "curr_date": curr_date,
            "metric_name": self.params["metric"],
            "metric_value": metric_value,
            "forecast": pred_df[self.params["target"]].to_numpy("float"),
            "actual": val_df[self.params["target"]].to_numpy(),
            "model_pickle": cloudpickle.dumps(model_fitted),
        }

        # Pass through native prediction intervals if the model provided them
        if "forecast_lower" in pred_df.columns and "forecast_upper" in pred_df.columns:
            result["forecast_lower"] = pred_df["forecast_lower"].to_numpy("float")
            result["forecast_upper"] = pred_df["forecast_upper"].to_numpy("float")

        return result
