from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import socketio
import eventlet
import base64
from datetime import datetime
import pymongo
import psycopg2
from dotenv import load_dotenv
import uuid
import torch

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize MongoDB connection
mongo_client = pymongo.MongoClient(os.getenv('MONGO_URI'))
mongo_db = mongo_client['object_detection']
tracked_objects = mongo_db['tracked_objects']

# Initialize PostgreSQL connection and create database if it doesn't exist
try:
    # First connect to default database to create our database
    conn = psycopg2.connect(
        dbname='postgres',
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        host=os.getenv('PG_HOST')
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Check if database exists
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'object_detection'")
    exists = cur.fetchone()
    
    if not exists:
        cur.execute('CREATE DATABASE object_detection')
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"Error setting up database: {e}")
    raise e

# Connect to our application database
try:
    pg_conn = psycopg2.connect(
        dbname=os.getenv('PG_DBNAME'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        host=os.getenv('PG_HOST')
    )
    pg_cursor = pg_conn.cursor()

    # Create table if not exists
    pg_cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_tracking (
            id SERIAL PRIMARY KEY,
            video_id UUID,
            track_id INTEGER,
            time_elapsed FLOAT,
            mongo_object_id TEXT
        )
    ''')
    pg_conn.commit()
except Exception as e:
    print(f"Error connecting to database: {e}")
    raise e

# Initialize YOLO model
print("Initializing YOLO model...")
print(f"CUDA available: {torch.cuda.is_available()}")
model = YOLO('yolov8n.pt')

# Export to TensorRT if CUDA is available
if torch.cuda.is_available():
    try:
        print("Attempting TensorRT export...")
        model.export(format='engine', device=0)
        model = YOLO('yolov8n.engine')
        print("TensorRT model loaded successfully")
    except Exception as e:
        print(f"TensorRT export failed: {e}")
        print("Falling back to regular model")
else:
    print("CUDA not available, using CPU model")

# Set device for inference
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Track statistics
track_stats = {}
THRESHOLD_SECONDS = 5  # Minimum time threshold for saving objects

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save the uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Reset track statistics
    global track_stats
    track_stats = {}
    
    # Start processing the video
    process_video(filepath, video_id)
    
    return jsonify({'message': 'Video uploaded successfully', 'filename': filename, 'video_id': video_id})

def process_video(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)
        
        # Update track statistics
        current_time = frame_count / fps
        for result in results:
            boxes = result.boxes
            for box in boxes:
                track_id = int(box.id.item()) if box.id is not None else None
                if track_id is not None:
                    if track_id not in track_stats:
                        track_stats[track_id] = {
                            'first_time': current_time,
                            'last_time': current_time,
                            'bboxes': []
                        }
                    else:
                        track_stats[track_id]['last_time'] = current_time
                        track_stats[track_id]['bboxes'].append(box.xyxy[0].cpu().numpy())
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame and track statistics to frontend
        sio.emit('frame', {
            'frame': frame_base64,
            'track_stats': {track_id: {
                'time_elapsed': stats['last_time'] - stats['first_time']
            } for track_id, stats in track_stats.items()}
        })
        
        frame_count += 1
        eventlet.sleep(0.03)
    
    cap.release()
    
    # Process and save tracked objects that meet the threshold
    for track_id, stats in track_stats.items():
        time_elapsed = stats['last_time'] - stats['first_time']
        if time_elapsed >= THRESHOLD_SECONDS:
            # Get the last bbox for cropping
            last_bbox = stats['bboxes'][-1]
            x1, y1, x2, y2 = map(int, last_bbox)
            
            # Read the last frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Crop the object
                cropped = frame[y1:y2, x1:x2]
                
                # Convert to base64 for MongoDB
                _, buffer = cv2.imencode('.jpg', cropped)
                cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Save to MongoDB
                mongo_result = tracked_objects.insert_one({
                    'video_id': video_id,
                    'track_id': track_id,
                    'time_elapsed': time_elapsed,
                    'image': cropped_base64
                })
                
                # Save to PostgreSQL
                pg_cursor.execute(
                    'INSERT INTO object_tracking (video_id, track_id, time_elapsed, mongo_object_id) VALUES (%s, %s, %s, %s)',
                    (video_id, track_id, time_elapsed, str(mongo_result.inserted_id))
                )
                pg_conn.commit()
    
    sio.emit('processing_complete', {'message': 'Video processing completed'})

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app) 