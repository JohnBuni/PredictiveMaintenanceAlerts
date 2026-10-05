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
        return psycopg.connect(self.database_url)

    def test_connection(self):
        """Test the database connection."""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()

        return result[0] == 1

    def create_tables(self):
        """Create the project tables if they do not already exist."""

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

    def upload_training_data(self, dataframe):
        """Upload a training DataFrame into PostgreSQL."""

        insert_query = """
        INSERT INTO training_data (
            trait,
            axis_1, axis_2, axis_3, axis_4,
            axis_5, axis_6, axis_7, axis_8,
            timestamp
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        records = []

        for _, row in dataframe.iterrows():
            records.append(
                (
                    row["Trait"],
                    row["Axis #1"],
                    row["Axis #2"],
                    row["Axis #3"],
                    row["Axis #4"],
                    row["Axis #5"],
                    row["Axis #6"],
                    row["Axis #7"],
                    row["Axis #8"],
                    row["Time"],
                )
            )

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(insert_query, records)

            connection.commit()

    def get_training_data(self):
        """Retrieve the training dataset from PostgreSQL."""

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
            dataframe = pd.read_sql(query, connection)

        return dataframe

    def clear_training_data(self):
        """Remove existing training records before a fresh upload."""

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "TRUNCATE TABLE training_data RESTART IDENTITY;"
                )

            connection.commit()