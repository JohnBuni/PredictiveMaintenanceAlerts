import pandas as pd


class AnomalyDetector:
    """Detect sustained Alert and Error conditions from regression residuals."""

    def __init__(self, regression_analyzer, thresholds, duration_threshold):
        self.regression = regression_analyzer
        self.thresholds = thresholds
        self.duration_threshold = duration_threshold

    def detect(self, test_df, training_df):
        """Detect sustained positive deviations for all eight axes."""
        df = test_df.copy()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        training_start = pd.to_datetime(
            training_df["timestamp"],
            utc=True
        ).min()

        # Use the same time origin that was used during model training.
        df["elapsed_seconds"] = (
            df["timestamp"] - training_start
        ).dt.total_seconds()

        events = []

        for axis in range(1, 9):
            column = f"axis_{axis}"
            model = self.regression.models[axis]

            # Calculate expected current and residual.
            predictions = model.predict(df[["elapsed_seconds"]])

            df[f"axis_{axis}_predicted"] = predictions
            df[f"axis_{axis}_residual"] = (
                df[column] - predictions
            )

            min_c = self.thresholds[axis]["MinC"]
            max_c = self.thresholds[axis]["MaxC"]

            # Error takes priority over Alert.
            conditions = []

            for residual in df[f"axis_{axis}_residual"]:
                if residual >= max_c:
                    conditions.append("Error")
                elif residual >= min_c:
                    conditions.append("Alert")
                else:
                    conditions.append(None)

            df[f"axis_{axis}_status"] = conditions

            events.extend(
                self._find_sustained_events(
                    df,
                    axis
                )
            )

        events_df = pd.DataFrame(events)

        return df, events_df

    def _find_sustained_events(self, df, axis):
        """Convert consecutive abnormal readings into sustained events."""
        status_column = f"axis_{axis}_status"
        residual_column = f"axis_{axis}_residual"

        events = []

        current_status = None
        start_index = None

        for i in range(len(df)):
            status = df.iloc[i][status_column]

            if status != current_status:
                if current_status is not None:
                    event = self._build_event(
                        df,
                        axis,
                        current_status,
                        start_index,
                        i - 1,
                        residual_column
                    )

                    if event is not None:
                        events.append(event)

                current_status = status
                start_index = i if status is not None else None

        # Handle an event that continues through the final record.
        if current_status is not None:
            event = self._build_event(
                df,
                axis,
                current_status,
                start_index,
                len(df) - 1,
                residual_column
            )

            if event is not None:
                events.append(event)

        return events

    def _build_event(
        self,
        df,
        axis,
        event_type,
        start_index,
        end_index,
        residual_column
    ):
        """Create an event only when its duration reaches T."""
        start_time = df.iloc[start_index]["timestamp"]
        end_time = df.iloc[end_index]["timestamp"]

        duration = (
            end_time - start_time
        ).total_seconds()

        if duration < self.duration_threshold:
            return None

        max_deviation = df.iloc[
            start_index:end_index + 1
        ][residual_column].max()

        return {
            "axis": axis,
            "event_type": event_type,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "max_deviation": max_deviation
        }