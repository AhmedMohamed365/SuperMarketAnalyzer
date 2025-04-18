import React, { useState, useEffect, useRef } from 'react';
import { Container, Box, Paper, Typography, Button, Grid, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Checkbox, FormControlLabel, Slider } from '@mui/material';
import { styled } from '@mui/material/styles';
import io from 'socket.io-client';

const VideoContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  textAlign: 'center',
}));

const VideoDisplay = styled('img')({
  maxWidth: '100%',
  maxHeight: '500px',
  marginTop: '20px',
});

const CanvasContainer = styled(Box)({
  position: 'relative',
  marginTop: '20px',
  width: '100%',
  height: '500px',
  backgroundColor: 'black',
});

const Canvas = styled('canvas')({
  position: 'absolute',
  top: 0,
  left: 0,
  zIndex: 10,
  width: '100%',
  height: '100%',
});

const DashboardContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  maxHeight: '300px',
  overflow: 'auto',
}));

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [currentFrame, setCurrentFrame] = useState('');
  const [processing, setProcessing] = useState(false);
  const [socket, setSocket] = useState(null);
  const [trackStats, setTrackStats] = useState({});
  const [drawingROI, setDrawingROI] = useState(false);
  const [roiPoints, setRoiPoints] = useState([]);
  const [useROI, setUseROI] = useState(false);
  const [threshold, setThreshold] = useState(5);
  const canvasRef = useRef(null);

  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('frame', (data) => {
      setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
      setTrackStats(data.track_stats || {});
    });

    newSocket.on('processing_complete', () => {
      setProcessing(false);
    });

    return () => newSocket.close();
  }, []);

  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
    setRoiPoints([]);
    setUseROI(false);
  };

  const handleDrawROI = () => {
    setDrawingROI(true);
    setRoiPoints([]);
    drawROI();
  };

  const handleCanvasClick = (e) => {
    if (!drawingROI || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setRoiPoints(prevPoints => [...prevPoints, { x, y }]);
    drawROI();
  };

  const drawROI = () => {
    if (!canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    // Draw the polygon
    if (roiPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(roiPoints[0].x, roiPoints[0].y);
      
      for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i].x, roiPoints[i].y);
      }
      
      if (roiPoints.length > 2) {
        ctx.closePath();
      }
      
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw the points
      roiPoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
      });
    }
  };

  const handleFinishROI = () => {
    setDrawingROI(false);
  };

  const handleClearROI = () => {
    setRoiPoints([]);
    drawROI();
  };

  const handleUpload = async () => {
    if (!videoFile) return;

    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('threshold', threshold.toString());
    
    // Add ROI points if they exist and ROI is enabled
    if (useROI && roiPoints.length > 0) {
      formData.append('roi_points', JSON.stringify(roiPoints));
    }

    try {
      setProcessing(true);
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      console.log('Upload successful:', data);
    } catch (error) {
      console.error('Error uploading video:', error);
      setProcessing(false);
    }
  };

  useEffect(() => {
    drawROI();
  }, [roiPoints]);

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          YOLO Video Object Detection
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <VideoContainer>
              <input
                accept="video/*"
                style={{ display: 'none' }}
                id="video-upload"
                type="file"
                onChange={handleFileChange}
              />
              <label htmlFor="video-upload">
                <Button
                  variant="contained"
                  component="span"
                  disabled={processing}
                >
                  Select Video
                </Button>
              </label>
              
              {videoFile && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body1">
                    Selected file: {videoFile.name}
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={useROI}
                        onChange={(e) => setUseROI(e.target.checked)}
                      />
                    }
                    label="Use Region of Interest (ROI)"
                  />
                  
                  {useROI && (
                    <>
                      <Typography variant="body1" sx={{ mt: 2 }}>
                        Draw Region of Interest
                      </Typography>
                      
                      <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center', gap: 1 }}>
                        <Button
                          variant="outlined"
                          onClick={handleDrawROI}
                          disabled={drawingROI}
                        >
                          Start Drawing
                        </Button>
                        <Button
                          variant="outlined"
                          onClick={handleFinishROI}
                          disabled={!drawingROI || roiPoints.length < 3}
                        >
                          Finish Drawing
                        </Button>
                        <Button
                          variant="outlined"
                          onClick={handleClearROI}
                          disabled={roiPoints.length === 0}
                        >
                          Clear
                        </Button>
                      </Box>
                      
                      <CanvasContainer>
                        <Canvas
                          ref={canvasRef}
                          width={800}
                          height={500}
                          onClick={handleCanvasClick}
                          style={{ 
                            cursor: drawingROI ? 'crosshair' : 'default' 
                          }}
                        />
                      </CanvasContainer>
                    </>
                  )}
                  
                  <Box sx={{ mt: 2 }}>
                    <Typography gutterBottom>
                      Time Threshold (seconds): {threshold}
                    </Typography>
                    <Slider
                      value={threshold}
                      onChange={(e, newValue) => setThreshold(newValue)}
                      min={1}
                      max={30}
                      step={1}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleUpload}
                    disabled={processing}
                    sx={{ mt: 2 }}
                  >
                    {processing ? 'Processing...' : 'Upload and Process'}
                  </Button>
                </Box>
              )}
            </VideoContainer>

            {currentFrame && (
              <VideoContainer>
                <Typography variant="h6" gutterBottom>
                  Live Detection Results
                </Typography>
                <VideoDisplay src={currentFrame} alt="Processed video frame" />
              </VideoContainer>
            )}
          </Grid>

          <Grid item xs={12} md={4}>
            <DashboardContainer>
              <Typography variant="h6" gutterBottom>
                Tracking Dashboard
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Track ID</TableCell>
                      <TableCell align="right">Time Elapsed (s)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(trackStats)
                      .sort(([, a], [, b]) => b.time_elapsed - a.time_elapsed)
                      .map(([trackId, stats]) => (
                        <TableRow key={trackId}>
                          <TableCell component="th" scope="row">
                            {trackId}
                          </TableCell>
                          <TableCell align="right">
                            {stats.time_elapsed.toFixed(2)}
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </DashboardContainer>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default App; 