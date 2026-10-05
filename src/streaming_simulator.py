class StreamingSimulator:
    """Simulate chronological robot data flow into PostgreSQL."""

    def __init__(self, database_manager):
        self.database = database_manager

    def stream(self, dataframe):
        """Upload timestamp-ordered test observations to PostgreSQL."""

        stream_df = (
            dataframe
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Clear previous simulation results for reproducible reruns.
        self.database.clear_stream_data()

        # PostgreSQL COPY preserves the ordered synthetic observations
        # without requiring hundreds of individual network transactions.
        self.database.upload_stream_data(stream_df)

        return len(stream_df)