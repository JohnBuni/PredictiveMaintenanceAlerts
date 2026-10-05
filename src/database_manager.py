import os

import pandas as pd
import psycopg
from dotenv import load_dotenv


class DatabaseManager:
    """Manage the PostgreSQL database used by the project."""

    def __init__(self):
        load_dotenv()
        self.database_url = os.getenv("DATABASE_URL")

        if not self.database_url:
            raise ValueError(
                "DATABASE_URL was not found. "
                "Make sure a .env file exists in the project root."
            )

    def connect(self):
        """Open and return a PostgreSQL connection."""
        return psycopg.connect(
            self.database_url,
            connect_timeout=15
        )

    def test_connection(self):
        """Test the database connection."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()

        return result[0] == 1

    def create_tables(self):
        """Create all required project tables."""

        training_table = """
        CREATE TABLE IF NOT EXISTS training_data (
            id SERIAL PRIMARY KEY,
            trait TEXT,
            axis_1 DOUBLE PRECISION,
            axis_2 DOUBLE PRECISION,
            axis_3 DOUBLE PRECISION,
            axis_4 DOUBLE PRECISION,
            axis_5 DOUBLE PRECISION,
            axis_6 DOUBLE PRECISION,
            axis_7 DOUBLE PRECISION,
            axis_8 DOUBLE PRECISION,
            timestamp TIMESTAMPTZ
        );
        """

        stream_table = """
        CREATE TABLE IF NOT EXISTS stream_data (
            id SERIAL PRIMARY KEY,
            trait TEXT,
            axis_1 DOUBLE PRECISION,
            axis_2 DOUBLE PRECISION,
            axis_3 DOUBLE PRECISION,
            axis_4 DOUBLE PRECISION,
            axis_5 DOUBLE PRECISION,
            axis_6 DOUBLE PRECISION,
            axis_7 DOUBLE PRECISION,
            axis_8 DOUBLE PRECISION,
            timestamp TIMESTAMPTZ
        );
        """

        events_table = """
        CREATE TABLE IF NOT EXISTS anomaly_events (
            id SERIAL PRIMARY KEY,
            axis INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            start_time TIMESTAMPTZ,
            end_time TIMESTAMPTZ,
            duration_seconds DOUBLE PRECISION,
            max_deviation DOUBLE PRECISION
        );
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(training_table)
                cursor.execute(stream_table)
                cursor.execute(events_table)

            connection.commit()

    def clear_training_data(self):
        """Remove previous training records."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "TRUNCATE TABLE training_data RESTART IDENTITY;"
                )

            connection.commit()

    def upload_training_data(self, dataframe):
        """Upload training data efficiently using PostgreSQL COPY."""

        copy_query = """
        COPY training_data (
            trait,
            axis_1, axis_2, axis_3, axis_4,
            axis_5, axis_6, axis_7, axis_8,
            timestamp
        )
        FROM STDIN
        """

        columns = [
            "Trait",
            "Axis #1",
            "Axis #2",
            "Axis #3",
            "Axis #4",
            "Axis #5",
            "Axis #6",
            "Axis #7",
            "Axis #8",
            "Time",
        ]

        with self.connect() as connection:
            with connection.cursor() as cursor:
                with cursor.copy(copy_query) as copy:
                    for row in dataframe[columns].itertuples(
                        index=False,
                        name=None
                    ):
                        copy.write_row(row)

            connection.commit()

    def get_training_data(self):
        """Retrieve training data from PostgreSQL."""

        query = """
        SELECT
            trait,
            axis_1, axis_2, axis_3, axis_4,
            axis_5, axis_6, axis_7, axis_8,
            timestamp
        FROM training_data
        ORDER BY timestamp;
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)

                rows = cursor.fetchall()

                columns = [
                    description.name
                    for description in cursor.description
                ]

        return pd.DataFrame(rows, columns=columns)

    def clear_stream_data(self):
        """Remove previous simulated streaming records."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "TRUNCATE TABLE stream_data RESTART IDENTITY;"
                )

            connection.commit()

    def upload_stream_data(self, dataframe):
        """Upload simulated stream efficiently using PostgreSQL COPY."""

        copy_query = """
        COPY stream_data (
            trait,
            axis_1, axis_2, axis_3, axis_4,
            axis_5, axis_6, axis_7, axis_8,
            timestamp
        )
        FROM STDIN
        """

        columns = [
            "trait",
            "axis_1",
            "axis_2",
            "axis_3",
            "axis_4",
            "axis_5",
            "axis_6",
            "axis_7",
            "axis_8",
            "timestamp",
        ]

        with self.connect() as connection:
            with connection.cursor() as cursor:
                with cursor.copy(copy_query) as copy:
                    for row in dataframe[columns].itertuples(
                        index=False,
                        name=None
                    ):
                        copy.write_row(row)

            connection.commit()

    def get_stream_data(self):
        """Retrieve simulated streaming data from PostgreSQL."""

        query = """
        SELECT
            trait,
            axis_1, axis_2, axis_3, axis_4,
            axis_5, axis_6, axis_7, axis_8,
            timestamp
        FROM stream_data
        ORDER BY timestamp;
        """

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)

                rows = cursor.fetchall()

                columns = [
                    description.name
                    for description in cursor.description
                ]

        return pd.DataFrame(rows, columns=columns)

    def clear_anomaly_events(self):
        """Remove previous anomaly-event records."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "TRUNCATE TABLE anomaly_events RESTART IDENTITY;"
                )

            connection.commit()

    def upload_anomaly_events(self, events_df):
        """Store detected Alert and Error events in PostgreSQL."""

        if events_df.empty:
            return

        copy_query = """
        COPY anomaly_events (
            axis,
            event_type,
            start_time,
            end_time,
            duration_seconds,
            max_deviation
        )
        FROM STDIN
        """

        columns = [
            "axis",
            "event_type",
            "start_time",
            "end_time",
            "duration_seconds",
            "max_deviation",
        ]

        with self.connect() as connection:
            with connection.cursor() as cursor:
                with cursor.copy(copy_query) as copy:
                    for row in events_df[columns].itertuples(
                        index=False,
                        name=None
                    ):
                        copy.write_row(row)

            connection.commit()