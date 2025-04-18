from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import socketio
import eventlet
import base64
import json
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

# Create directories if they don't exist
UPLOAD_FOLDER = 'uploads'
ROI_FOLDER = 'roi_masks'
for folder in [UPLOAD_FOLDER, ROI_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

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

# Default threshold for tracking
DEFAULT_THRESHOLD_SECONDS = 5

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
    
    # Process ROI if provided
    roi_mask = None
    if 'roi_points' in request.form:
        try:
            roi_points = json.loads(request.form['roi_points'])
            if roi_points and len(roi_points) >= 3:
                # Get video dimensions
                cap = cv2.VideoCapture(filepath)
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Calculate scaling factors
                canvas_width = 800  # Canvas width from frontend
                canvas_height = 500  # Canvas height from frontend
                scale_x = video_width / canvas_width
                scale_y = video_height / canvas_height
                
                # Scale ROI points to match video dimensions
                scaled_points = []
                for point in roi_points:
                    scaled_x = int(point['x'] * scale_x)
                    scaled_y = int(point['y'] * scale_y)
                    scaled_points.append((scaled_x, scaled_y))
                
                # Create ROI mask
                roi_mask = create_roi_mask(filepath, scaled_points)
                # Save ROI mask
                roi_mask_path = os.path.join(ROI_FOLDER, f"{video_id}_mask.png")
                cv2.imwrite(roi_mask_path, roi_mask)
        except Exception as e:
            print(f"Error processing ROI: {e}")
    
    # Get custom threshold if provided
    threshold_seconds = DEFAULT_THRESHOLD_SECONDS
    if 'threshold' in request.form:
        try:
            threshold_seconds = float(request.form['threshold'])
        except ValueError:
            print(f"Invalid threshold value: {request.form['threshold']}, using default")
    
    # Reset track statistics
    global track_stats
    track_stats = {}
    
    # Start processing the video
    process_video(filepath, video_id, roi_mask, threshold_seconds)
    
    return jsonify({
        'message': 'Video uploaded successfully', 
        'filename': filename, 
        'video_id': video_id,
        'threshold': threshold_seconds
    })

def create_roi_mask(video_path, roi_points):
    # Extract first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Could not read video file")
    
    # Create empty mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Convert ROI points to numpy array
    roi_np = np.array(roi_points, dtype=np.int32)
    
    # Draw filled polygon on mask
    cv2.fillPoly(mask, [roi_np], 255)
    
    return mask

def is_point_in_roi(point, roi_mask):
    if roi_mask is None:
        return True
    
    x, y = int(point[0]), int(point[1])
    # Check if the point is within mask boundaries
    h, w = roi_mask.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return roi_mask[y, x] > 0
    return False

def process_video(video_path, video_id, roi_mask=None, threshold_seconds=DEFAULT_THRESHOLD_SECONDS):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    # Dictionary to track which IDs have exceeded threshold
    exceeded_threshold = {}
    
    # Dictionary to store track stats for each ID
    track_stats = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection with tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)
        
        # Update track statistics
        current_time = frame_count / fps
        
        # Make a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                track_id = int(box.id.item()) if box.id is not None else None
                
                if track_id is not None:
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Calculate center point of the box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Skip if the detection is outside ROI
                    if not is_point_in_roi((center_x, center_y), roi_mask):
                        continue
                    
                    # Update track stats
                    if track_id not in track_stats:
                        track_stats[track_id] = {
                            'first_time': current_time,
                            'last_time': current_time,
                            'bboxes': [xyxy.tolist()]
                        }
                    else:
                        track_stats[track_id]['last_time'] = current_time
                        track_stats[track_id]['bboxes'].append(xyxy.tolist())
                    
                    # Calculate time elapsed for this track
                    time_elapsed = track_stats[track_id]['last_time'] - track_stats[track_id]['first_time']
                    
                    # Check if this track has exceeded the threshold
                    if time_elapsed >= threshold_seconds:
                        exceeded_threshold[track_id] = True
                    
                    # Determine color: red if exceeded threshold, blue otherwise
                    color = (255, 0, 0) if track_id in exceeded_threshold else (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID and time
                    label = f"ID:{track_id} {time_elapsed:.1f}s"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ROI contour if exists
        if roi_mask is not None:
            # Find contours in the mask
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame and track statistics to frontend
        sio.emit('frame', {
            'frame': frame_base64,
            'track_stats': {str(track_id): {
                'time_elapsed': stats['last_time'] - stats['first_time'],
                'exceeded_threshold': track_id in exceeded_threshold
            } for track_id, stats in track_stats.items()}
        })
        
        frame_count += 1
        eventlet.sleep(0.03)
    
    cap.release()
    
    # Process and save tracked objects that meet the threshold
    for track_id, stats in track_stats.items():
        time_elapsed = stats['last_time'] - stats['first_time']
        if time_elapsed >= threshold_seconds:
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