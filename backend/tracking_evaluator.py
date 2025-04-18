# tracking_evaluator.py
import cv2
import numpy as np
from collections import defaultdict

class TrackingEvaluator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.id_switches = 0
        self.track_fragmentations = 0
        self.track_durations = defaultdict(list)
        self.current_tracks = {}
        self.frame_count = 0
    
    def update(self, tracks):
        """
        Update metrics for current frame
        tracks: dict of {track_id: bbox}
        """
        self.frame_count += 1
        
        # Check for ID switches and track fragmentations
        current_ids = set(tracks.keys())
        prev_ids = set(self.current_tracks.keys())
        
        # Track fragmentations (disappearing tracks)
        disappeared = prev_ids - current_ids
        for track_id in disappeared:
            self.track_fragmentations += 1
            duration = self.frame_count - self.current_tracks[track_id]['start_frame']
            self.track_durations[track_id].append(duration)
        
        # New or reappeared tracks
        appeared = current_ids - prev_ids
        for track_id in appeared:
            if track_id in self.track_durations:
                self.id_switches += 1
            self.current_tracks[track_id] = {
                'start_frame': self.frame_count,
                'bbox': tracks[track_id]
            }
        
        # Update continuing tracks
        continuing = current_ids & prev_ids
        for track_id in continuing:
            self.current_tracks[track_id]['bbox'] = tracks[track_id]
    
    def get_metrics(self):
        """Return evaluation metrics"""
        avg_duration = np.mean([max(durations) for durations in self.track_durations.values()]) if self.track_durations else 0
        
        return {
            'id_switches': self.id_switches,
            'fragmentations': self.track_fragmentations,
            'avg_track_duration': avg_duration,
            'stability_score': self._calculate_stability()
        }
    
    def _calculate_stability(self):
        """Calculate overall tracking stability score"""
        if not self.track_durations:
            return 0.0
        
        # Penalize ID switches and fragmentations while rewarding longer tracks
        avg_duration = np.mean([max(durations) for durations in self.track_durations.values()])
        total_tracks = len(self.track_durations)
        
        stability = (avg_duration / self.frame_count) * (1.0 / (1 + self.id_switches/total_tracks))
        return stability