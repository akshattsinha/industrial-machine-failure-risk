
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATABASE_PATH = BASE_DIR / "machine_predictions.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                machine_type TEXT NOT NULL,

                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,

                logistic_probability REAL NOT NULL,
                pca_probability REAL NOT NULL,
                random_forest_probability REAL NOT NULL,

                logistic_risk TEXT NOT NULL,
                pca_risk TEXT NOT NULL,
                random_forest_risk TEXT NOT NULL

            )
        """)

        connection.commit()


def save_prediction(machine, predictions):

    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO predictions (

                machine_type,

                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,

                logistic_probability,
                pca_probability,
                random_forest_probability,

                logistic_risk,
                pca_risk,
                random_forest_risk

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            machine.machine_type,

            machine.air_temperature,
            machine.process_temperature,
            machine.rotational_speed,
            machine.torque,
            machine.tool_wear,

            predictions["logistic_regression"]["failure_probability"],
            predictions["pca_logistic_regression"]["failure_probability"],
            predictions["random_forest"]["failure_probability"],

            predictions["logistic_regression"]["risk_level"],
            predictions["pca_logistic_regression"]["risk_level"],
            predictions["random_forest"]["risk_level"]

        ))

        connection.commit()

        return cursor.lastrowid


def get_prediction_history(limit=50):

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


def get_prediction_analytics():

    with get_connection() as connection:

        total = connection.execute("""
            SELECT COUNT(*) FROM predictions
        """).fetchone()[0]

        high_risk = connection.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE random_forest_risk = 'HIGH'
        """).fetchone()[0]

        medium_risk = connection.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE random_forest_risk = 'MEDIUM'
        """).fetchone()[0]

        low_risk = connection.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE random_forest_risk = 'LOW'
        """).fetchone()[0]

        average_probability = connection.execute("""
            SELECT AVG(random_forest_probability)
            FROM predictions
        """).fetchone()[0]

        return {
            "total_predictions": total,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
            "average_failure_probability": (
                round(average_probability, 4)
                if average_probability is not None
                else None
            )
        }
