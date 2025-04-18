import os
from pymongo import MongoClient
import base64
from datetime import datetime
import cv2

class MongoHandler:
    def __init__(self):
        # Connect to MongoDB
        self.client = MongoClient(os.getenv('MONGO_URI'))
        self.db = self.client['object_tracking']
        self.collection = self.db['tracked_objects']
    
    def save_tracked_object(self ,frame, bbox):
        """
        Save tracked object image to MongoDB
        
        Args:
            video_id (str): Unique video identifier
            track_id (int): Track identifier
            frame (numpy.ndarray): Frame containing the object
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            time_elapsed (float): Time elapsed in seconds
        
        Returns:
            str: MongoDB document ID
        """
        # Crop the object from the frame
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', cropped)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create document
        document = {
            'image': image_base64,
            
        }
        
        # Insert into MongoDB
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close() 