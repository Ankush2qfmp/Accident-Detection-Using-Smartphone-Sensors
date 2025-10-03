import sqlite3
import os

# Get database file from environment or use default
DATABASE_FILE = os.getenv('DATABASE_FILE', 'suraksha.db')

def fix_database():
    print(f"Fixing database schema in {DATABASE_FILE}...")
    
    # Connect to the database
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Check if sensor_logs table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sensor_logs'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        # Get column info
        cursor.execute(f"PRAGMA table_info(sensor_logs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Check and add missing columns
        missing_columns = []
        required_columns = [
            'acceleration_x', 'acceleration_y', 'acceleration_z', 'acceleration_magnitude',
            'gyroscope_x', 'gyroscope_y', 'gyroscope_z', 'gyroscope_magnitude',
            'latitude', 'longitude', 'anomaly_score'
        ]
        
        for col in required_columns:
            if col not in columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"Adding missing columns to sensor_logs table: {missing_columns}")
            
            # Add missing columns
            try:
                for col in missing_columns:
                    cursor.execute(f"ALTER TABLE sensor_logs ADD COLUMN {col} REAL")
                conn.commit()
                print("Database schema updated successfully!")
            except sqlite3.Error as e:
                print(f"Error updating database schema: {e}")
        else:
            print("All required columns already exist in sensor_logs table.")
    else:
        print("sensor_logs table doesn't exist. Creating it...")
        try:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                acceleration_x REAL,
                acceleration_y REAL,
                acceleration_z REAL,
                acceleration_magnitude REAL,
                gyroscope_x REAL,
                gyroscope_y REAL,
                gyroscope_z REAL,
                gyroscope_magnitude REAL,
                latitude REAL,
                longitude REAL,
                anomaly_score REAL
            )
            ''')
            conn.commit()
            print("sensor_logs table created successfully!")
        except sqlite3.Error as e:
            print(f"Error creating sensor_logs table: {e}")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    fix_database()
