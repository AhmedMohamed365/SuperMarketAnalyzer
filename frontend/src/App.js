import React, { useState, useEffect } from 'react';
import { Container, Box, Paper, Typography, Button, Grid, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
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
  };

  const handleUpload = async () => {
    if (!videoFile) return;

    const formData = new FormData();
    formData.append('video', videoFile);

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
                    {Object.entries(trackStats).map(([trackId, stats]) => (
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