import matplotlib.pyplot as plt
import pandas as pd


class MaintenanceVisualizer:
    """Visualize regression predictions and detected maintenance events."""

    def plot_detection_results(self, detected_df, events_df):
        """Plot observed currents, predictions, and Alert/Error events."""

        df = detected_df.copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        # Create a readable time axis for visualization only.
        # Regression calculations still use the original elapsed_seconds.
        df["stream_seconds"] = (
            df["timestamp"] - df["timestamp"].min()
        ).dt.total_seconds()

        fig, axes = plt.subplots(
            4,
            2,
            figsize=(16, 18)
        )

        axes = axes.flatten()

        for axis in range(1, 9):
            ax = axes[axis - 1]

            current_column = f"axis_{axis}"
            predicted_column = f"axis_{axis}_predicted"
            status_column = f"axis_{axis}_status"

            # Plot observed current.
            ax.plot(
                df["stream_seconds"],
                df[current_column],
                label="Observed",
                alpha=0.7
            )

            # Plot regression prediction.
            ax.plot(
                df["stream_seconds"],
                df[predicted_column],
                label="Regression Prediction",
                linewidth=2
            )

            # Show individual Alert threshold crossings.
            alert_mask = df[status_column] == "Alert"

            ax.scatter(
                df.loc[alert_mask, "stream_seconds"],
                df.loc[alert_mask, current_column],
                marker="^",
                s=70,
                label="Alert"
            )

            # Show individual Error threshold crossings.
            error_mask = df[status_column] == "Error"

            ax.scatter(
                df.loc[error_mask, "stream_seconds"],
                df.loc[error_mask, current_column],
                marker="X",
                s=80,
                label="Error"
            )

            # Annotate only sustained events that satisfy T.
            axis_events = events_df[
                events_df["axis"] == axis
            ]

            for _, event in axis_events.iterrows():
                start_time = pd.to_datetime(
                    event["start_time"],
                    utc=True
                )

                end_time = pd.to_datetime(
                    event["end_time"],
                    utc=True
                )

                event_mask = (
                    (df["timestamp"] >= start_time)
                    & (df["timestamp"] <= end_time)
                )

                if event_mask.any():
                    event_x = df.loc[
                        event_mask,
                        "stream_seconds"
                    ].mean()

                    event_y = df.loc[
                        event_mask,
                        current_column
                    ].max()

                    ax.annotate(
                        (
                            f'{event["event_type"]}\n'
                            f'{event["duration_seconds"]:.2f} s'
                        ),
                        xy=(event_x, event_y),
                        xytext=(0, 20),
                        textcoords="offset points",
                        ha="center",
                        arrowprops={
                            "arrowstyle": "->"
                        }
                    )

            ax.set_title(
                f"Axis #{axis}: Predictive Maintenance Detection"
            )

            ax.set_xlabel(
                "Seconds Since Stream Start"
            )

            ax.set_ylabel("Current")

            ax.grid(alpha=0.25)

            # Remove duplicate legend entries.
            handles, labels = (
                ax.get_legend_handles_labels()
            )

            unique = dict(zip(labels, handles))

            ax.legend(
                unique.values(),
                unique.keys(),
                fontsize=8
            )

        fig.suptitle(
            "Regression-Based Predictive Maintenance Alerts",
            fontsize=16
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.97]
        )

        return fig