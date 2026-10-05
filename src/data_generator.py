import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SyntheticDataGenerator:
    """Generate synthetic testing data from training-data characteristics."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def generate(self, training_df, n_samples=300):
        """Generate synthetic test data using training-data characteristics."""
        rng = np.random.default_rng(self.random_state)

        axis_columns = [f"axis_{i}" for i in range(1, 9)]
        training_values = training_df[axis_columns].copy()

        # Learn scaling parameters from training data only.
        self.standard_scaler.fit(training_values)
        self.minmax_scaler.fit(training_values)

        # Bootstrap complete rows to preserve realistic relationships
        # between robot state and the eight current axes.
        sampled_indices = rng.choice(
            training_df.index,
            size=n_samples,
            replace=True
        )

        synthetic = training_df.loc[
            sampled_indices,
            ["trait"] + axis_columns
        ].reset_index(drop=True).copy()

        # Add small variation to active currents so the synthetic
        # observations are not exact copies of training rows.
        for column in axis_columns:
            std = training_df[column].std()

            noise = rng.normal(
                loc=0,
                scale=std * 0.05,
                size=n_samples
            )

            active = synthetic[column] > 0

            synthetic.loc[active, column] += noise[active]

            # Current measurements cannot be negative.
            synthetic[column] = synthetic[column].clip(lower=0)

        synthetic["trait"] = "SYNTHETIC"

        # Determine the normal sampling interval from training timestamps.
        timestamps = pd.to_datetime(
            training_df["timestamp"],
            utc=True
        )

        median_interval = (
            timestamps
            .sort_values()
            .diff()
            .dt.total_seconds()
            .dropna()
            .median()
        )

        # Begin synthetic observations immediately after training data.
        start_time = (
            timestamps.max()
            + pd.Timedelta(seconds=median_interval)
        )

        synthetic["timestamp"] = [
            start_time
            + pd.Timedelta(seconds=median_interval * i)
            for i in range(n_samples)
        ]

        # Standardize using parameters learned from training data.
        standardized = self.standard_scaler.transform(
            synthetic[axis_columns]
        )

        # Normalize using parameters learned from training data.
        normalized = self.minmax_scaler.transform(
            synthetic[axis_columns]
        )

        # Keep transformed values so the preprocessing is explicit
        # and reproducible for the assignment.
        for i, column in enumerate(axis_columns):
            synthetic[f"{column}_zscore"] = standardized[:, i]
            synthetic[f"{column}_normalized"] = normalized[:, i]

        return synthetic