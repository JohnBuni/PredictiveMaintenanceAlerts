import pandas as pd
from sklearn.linear_model import LinearRegression


class RegressionAnalyzer:
    """Train and analyze linear regression models for robot current axes."""

    def __init__(self):
        self.models = {}
        self.results = {}

    def prepare_data(self, dataframe):
        """Convert timestamps into elapsed seconds for regression."""
        df = dataframe.copy()

        # Make sure timestamps are proper datetime values.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Use elapsed seconds as the numeric Time variable.
        start_time = df["timestamp"].min()
        df["elapsed_seconds"] = (
            df["timestamp"] - start_time
        ).dt.total_seconds()

        return df

    def fit_models(self, dataframe):
        """Train one Time -> Axis regression model for each of the 8 axes."""
        df = self.prepare_data(dataframe)

        # Scikit-learn expects the feature to be a 2D array.
        X = df[["elapsed_seconds"]]

        for axis in range(1, 9):
            column = f"axis_{axis}"
            y = df[column]

            model = LinearRegression()
            model.fit(X, y)

            # Generate fitted values and residuals for later analysis.
            predictions = model.predict(X)
            residuals = y.to_numpy() - predictions

            self.models[axis] = model

            self.results[axis] = {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "predictions": predictions,
                "residuals": residuals,
                "r_squared": model.score(X, y),
            }

        return df

    def get_model_summary(self):
        """Return slope, intercept, and R-squared for all models."""
        summary = []

        for axis, result in self.results.items():
            summary.append(
                {
                    "Axis": f"Axis #{axis}",
                    "Slope": result["slope"],
                    "Intercept": result["intercept"],
                    "R²": result["r_squared"],
                }
            )

        return pd.DataFrame(summary)