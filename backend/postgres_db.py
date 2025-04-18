import os
import psycopg2
from psycopg2.extras import execute_values

class PostgresHandler:
    def __init__(self):
        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            dbname=os.getenv('PG_DBNAME'),
            user=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD'),
            host=os.getenv('PG_HOST')
        )
        self.cursor = self.conn.cursor()
        
        # Create table if not exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_tracking (
                id SERIAL PRIMARY KEY,
                video_id UUID,
                track_id INTEGER,
                time_elapsed FLOAT,
                mongo_object_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_tracking_data(self, video_id, track_id, time_elapsed, mongo_object_id):
        """
        Save tracking metadata to PostgreSQL
        
        Args:
            video_id (str): Unique video identifier
            track_id (int): Track identifier
            time_elapsed (float): Time elapsed in seconds
            mongo_object_id (str): MongoDB document ID
        """
        self.cursor.execute(
            'INSERT INTO object_tracking (video_id, track_id, time_elapsed, mongo_object_id) VALUES (%s, %s, %s, %s)',
            (video_id, track_id, time_elapsed, mongo_object_id)
        )
        self.conn.commit()
    
    def close(self):
        """Close PostgreSQL connection"""
        self.cursor.close()
        self.conn.close() 