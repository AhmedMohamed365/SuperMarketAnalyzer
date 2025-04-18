# YOLO Video Object Detection Application

This application provides a web interface for uploading videos and performing real-time object detection using YOLO (You Only Look Once) with tracking capabilities.

## Features

- Video upload through a web interface
- Real-time object detection using YOLOv8
- Object tracking across frames
- Live visualization of detection results
- Modern and responsive UI

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

## Setup

### Backend Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. The YOLO model will be automatically downloaded on first run.

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

1. Start the backend server:
```bash
python backend/app.py
```

2. In a new terminal, start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Click the "Select Video" button to choose a video file
2. Click "Upload and Process" to start the object detection
3. The processed video with detection results will be displayed in real-time

## Notes

- The application supports common video formats (MP4, AVI, etc.)
- Processing speed depends on your hardware and video resolution
- The YOLO model will be downloaded automatically on first run 