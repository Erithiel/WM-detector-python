import cv2
import numpy as np
import time
import requests
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import deque
import logging


logger = logging.getLogger(__name__)


@dataclass
class MachineState:
    """Represents the state of a washing machine"""
    machine_id: int
    is_running: bool
    last_state_change: float
    motion_history: deque
    circle_roi: Tuple[int, int, int, int]  # x, y, width, height

class WashingMachineDetector:
    """Detects washing machine state using motion detection on circular indicators"""

    def __init__(
        self,
        camera_source: int = 0,
        backend_url: str = "http://localhost:8000/api/machine-state",
        motion_threshold: float = 0.08,
        state_change_delay: float = 3.0,
        motion_history_size: int = 30,
        http_post: Optional[Callable[..., Any]] = None,
        request_timeout_s: float = 5.0,
        disable_backend: bool = True,
    ):
        """
        Initialize the washing machine detector

        Args:
            camera_source: Camera index or video file path
            backend_url: Backend endpoint URL for state updates
            motion_threshold: Threshold for detecting motion (0-1)
            state_change_delay: Seconds to wait before confirming state change
            motion_history_size: Number of frames to keep in motion history
            http_post: Injectable POST function for mocking (defaults to requests.post)
            request_timeout_s: Timeout for backend calls
            disable_backend: If True, skip any HTTP calls (dry-run mode)
        """
        self.camera_source = camera_source
        self.backend_url = backend_url
        self.motion_threshold = motion_threshold
        self.state_change_delay = state_change_delay
        self.motion_history_size = motion_history_size

        self.http_post = http_post or requests.post
        self.request_timeout_s = request_timeout_s
        self.disable_backend = disable_backend

        self.cap = None
        self.machines: List[MachineState] = []
        self.setup_complete = False
        self.prev_frame = None

    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            logger.error("Cannot open camera %s", self.camera_source)
            return False

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        logger.info("Camera initialized successfully")
        return True

    def setup_machine_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Interactive setup to define circular ROIs for each machine

        Args:
            frame: The current video frame

        Returns:
            List of ROI coordinates (x, y, width, height) for each circle
        """
        logger.info("=== Machine Circle Setup ===")
        logger.info("Instructions:")
        logger.info("1. Draw rectangles around each circular indicator (left to right)")
        logger.info("2. Press 'c' to confirm current selection")
        logger.info("3. Press 'd' to delete last selection")
        logger.info("4. Press 'f' to finish setup")
        logger.info("5. Press 'q' to quit")

        circles = []
        drawing = False
        start_point = None
        current_rect = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, current_rect

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    current_rect = (start_point[0], start_point[1],
                                   x - start_point[0], y - start_point[1])

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if start_point:
                    current_rect = (start_point[0], start_point[1],
                                   x - start_point[0], y - start_point[1])

        cv2.namedWindow('Setup Machine Circles')
        cv2.setMouseCallback('Setup Machine Circles', mouse_callback)

        while True:
            display_frame = frame.copy()

            # Draw confirmed circles
            for idx, (cx, cy, w, h) in enumerate(circles):
                cv2.rectangle(display_frame, (cx, cy), (cx + w, cy + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Machine {idx + 1}", (cx, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw current rectangle being drawn
            if current_rect:
                x, y, w, h = current_rect
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(display_frame, f"Machines defined: {len(circles)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Setup Machine Circles', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and current_rect:
                x, y, w, h = current_rect
                if w > 0 and h > 0:
                    circles.append((x, y, w, h))
                    logger.info("Machine %s circle defined at (%s, %s, %s, %s)", len(circles), x, y, w, h)
                    current_rect = None

            elif key == ord('d') and circles:
                removed = circles.pop()
                logger.info("Removed last circle: %s", removed)

            elif key == ord('f') and circles:
                logger.info("Setup complete! %s machines defined.", len(circles))
                break

            elif key == ord('q'):
                logger.warning("Setup cancelled")
                circles = []
                break

        cv2.destroyWindow('Setup Machine Circles')
        return circles

    def detect_motion_in_roi(self, frame: np.ndarray, prev_frame: np.ndarray,
                            roi: Tuple[int, int, int, int]) -> float:
        """
        Detect motion within a specific ROI

        Args:
            frame: Current frame
            prev_frame: Previous frame
            roi: Region of interest (x, y, width, height)

        Returns:
            Motion score (0-1)
        """
        x, y, w, h = roi

        # Ensure ROI is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return 0.0

        # Extract ROI from both frames
        current_roi = frame[y:y+h, x:x+w]
        prev_roi = prev_frame[y:y+h, x:x+w]

        # Convert to grayscale
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise.
        # Make kernel size relative to ROI size to keep similar smoothing across scales.
        # Base it on the smaller ROI dimension.
        min_dim = max(1, min(w, h))
        # Roughly 1/10 of min dimension, forced odd and within [3, 31]
        k = max(3, min(31, (min_dim // 10) * 2 + 1))
        current_blur = cv2.GaussianBlur(current_gray, (k, k), 0)
        prev_blur = cv2.GaussianBlur(prev_gray, (k, k), 0)

        # Calculate absolute difference
        frame_diff = cv2.absdiff(current_blur, prev_blur)

        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Create a circular mask inside the ROI so motion is measured relative to the indicator circle,
        # not the full rectangle. This makes scores comparable for small vs large circles.
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        radius = max(1, min(w, h) // 2 - 1)
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)

        # Count changed pixels only inside the circular mask and normalize by mask area
        masked_changes = cv2.bitwise_and(thresh, thresh, mask=mask)
        changed_pixels = int(np.count_nonzero(masked_changes))
        mask_area = int(np.count_nonzero(mask))
        if mask_area == 0:
            return 0.0
#asdasd
        motion_score = changed_pixels / float(mask_area)

        return motion_score

    def detect_rotation_in_circle(self, frame: np.ndarray, prev_frame: np.ndarray,
                                  roi: Tuple[int, int, int, int]) -> float:
        """
        Rotation-aware motion detection inside the circular ROI.

        Converts the ROI to polar coordinates so rotation becomes a horizontal shift
        and estimates the angular shift between frames via circular cross-correlation.

        Returns a normalized motion score in [0, 1] roughly proportional to the
        absolute angular change per frame. The scale is designed so ~180°/frame
        maps to ~1.0 (actual shifts are usually far smaller), making thresholds
        comparable to the prior implementation (~0.04 default).
        """
        x, y, w, h = roi

        # Clamp ROI to frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w <= 0 or h <= 0:
            return 0.0

        # Extract grayscale ROIs
        cur = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        prev = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        # Determine circle center and radius within ROI
        cx, cy = w // 2, h // 2
        R = max(1, min(w, h) // 2 - 1)
        if R <= 2:
            return 0.0

        # Light blur relative to ROI size (reuse prior heuristic)
        min_dim = max(1, min(w, h))
        k = max(3, min(31, (min_dim // 10) * 2 + 1))
        cur_blur = cv2.GaussianBlur(cur, (k, k), 0)
        prev_blur = cv2.GaussianBlur(prev, (k, k), 0)

        # Edge/gradient magnitude to reduce illumination sensitivity
        cur_gx = cv2.Sobel(cur_blur, cv2.CV_32F, 1, 0, ksize=3)
        cur_gy = cv2.Sobel(cur_blur, cv2.CV_32F, 0, 1, ksize=3)
        prev_gx = cv2.Sobel(prev_blur, cv2.CV_32F, 1, 0, ksize=3)
        prev_gy = cv2.Sobel(prev_blur, cv2.CV_32F, 0, 1, ksize=3)
        cur_mag = cv2.magnitude(cur_gx, cur_gy)
        prev_mag = cv2.magnitude(prev_gx, prev_gy)

        # Build polar images: dimensions (angles, radii)
        # Choose angular samples; 180–360 is enough. Use 360 for clarity.
        N_theta = 360
        # Radial samples limited to the radius
        N_r = R

        # OpenCV warpPolar expects center in full-image coords; we operate on ROI, so center is (cx, cy)
        flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS

        cur_polar = cv2.warpPolar(cur_mag, (N_theta, N_r), (cx, cy), R, flags)
        prev_polar = cv2.warpPolar(prev_mag, (N_theta, N_r), (cx, cy), R, flags)

        # Angular signatures by averaging over a ring band to avoid center/border artifacts
        r0 = max(1, int(0.55 * R))
        r1 = max(r0 + 1, int(0.90 * R))
        r1 = min(r1, N_r)
        if r1 <= r0:
            return 0.0

        cur_sig = np.mean(cur_polar[r0:r1, :], axis=0).astype(np.float32)
        prev_sig = np.mean(prev_polar[r0:r1, :], axis=0).astype(np.float32)

        # Normalize signatures to zero-mean, unit-variance
        def normalize(sig: np.ndarray) -> np.ndarray:
            s = sig - np.mean(sig)
            std = float(np.std(s))
            return s / std if std > 1e-6 else s * 0.0

        a = normalize(cur_sig)
        b = normalize(prev_sig)

        # If both are near-zero (flat), no motion info
        if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
            return 0.0

        # Circular cross-correlation via FFT
        Fa = np.fft.rfft(a)
        Fb = np.fft.rfft(b)
        cc = np.fft.irfft(Fa * np.conj(Fb), n=a.shape[0])

        # Normalize correlation by vector norms to get confidence in [-1,1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 1e-6:
            cc_norm = cc / denom
        else:
            cc_norm = cc

        # Find best shift (wrap-around index)
        k_max = int(np.argmax(cc_norm))
        peak = float(cc_norm[k_max])

        # Convert shift index to degrees (wrap so shortest rotation)
        # Shifts > N/2 correspond to negative rotation
        N = a.shape[0]
        if k_max > N // 2:
            k_shift = k_max - N
        else:
            k_shift = k_max

        deg_per_bin = 360.0 / float(N)
        delta_deg = k_shift * deg_per_bin

        # Motion score: absolute angular change per frame, normalized by 180°
        # Modulate by correlation confidence clipped to [0,1]
        confidence = max(0.0, min(1.0, (peak + 1.0) / 2.0))  # map [-1,1] -> [0,1]
        score = min(1.0, abs(delta_deg) / 180.0) * confidence
        return float(score)

    def send_state_update(self, machine_id: int, is_running: bool) -> bool:
        """
        Send machine state update to backend

        Args:
            machine_id: ID of the machine (1-indexed)
            is_running: Whether the machine is running

        Returns:
            True if successful, False otherwise
        """
        # If backend updates are disabled, do nothing and report success
        if getattr(self, "disable_backend", False):
            logger.debug(
                "Backend disabled; skipping state update for machine %s -> %s",
                machine_id,
                "RUNNING" if is_running else "STOPPED",
            )
            return True

        payload = {
            "machine_id": machine_id,
            "state": "running" if is_running else "stopped",
            "timestamp": time.time(),
        }

        try:
            logger.debug("POST %s payload=%s", self.backend_url, payload)

            response = self.http_post(
                self.backend_url,
                json=payload,
                timeout=self.request_timeout_s,
            )

            if getattr(response, "status_code", None) == 200:
                logger.info("Machine %s state updated: %s", machine_id, payload["state"].upper())
                return True

            logger.warning(
                "Failed to update machine %s: HTTP %s",
                machine_id,
                getattr(response, "status_code", "<?>"),
            )
            return False

        except requests.exceptions.RequestException:
            logger.exception("Network error sending update for machine %s", machine_id)
            return False
        except Exception:
            logger.exception("Unexpected error sending update for machine %s", machine_id)
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and update machine states

        Args:
            frame: Current video frame

        Returns:
            Annotated frame for display
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame

        current_time = time.time()
        display_frame = frame.copy()

        # Process each machine
        for machine in self.machines:
            # Detect rotation-aware motion in this machine's circle
            motion_score = self.detect_rotation_in_circle(frame, self.prev_frame, machine.circle_roi)

            # Add to motion history
            machine.motion_history.append(motion_score)

            # Calculate average motion over history
            avg_motion = float(np.mean(machine.motion_history)) if machine.motion_history else 0.0
            # Determine if machine should be running based on motion
            motion_detected = avg_motion > self.motion_threshold

            # Check if state has changed
            if motion_detected != machine.is_running:
                time_since_change = current_time - machine.last_state_change

                # Only change state if enough time has passed (debounce)
                if time_since_change > self.state_change_delay:
                    machine.is_running = motion_detected
                    machine.last_state_change = current_time

                    logger.info(
                        "State change confirmed for machine %s -> %s (avg_motion=%.3f)",
                        machine.machine_id,
                        "RUNNING" if machine.is_running else "STOPPED",
                        avg_motion,
                    )

                    # Send update to backend
                    self.send_state_update(machine.machine_id, machine.is_running)

            # Draw visualization
            x, y, w, h = machine.circle_roi
            color = (0, 255, 0) if machine.is_running else (0, 0, 255)

            # Draw rectangle around circle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            # Draw machine info
            status = "RUNNING" if machine.is_running else "STOPPED"
            cv2.putText(display_frame, f"M{machine.machine_id}: {status}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw motion indicator
            motion_bar_length = int(avg_motion * 100)
            cv2.rectangle(display_frame, (x, y + h + 5),
                         (x + motion_bar_length, y + h + 15), color, -1)
            cv2.putText(display_frame, f"{avg_motion:.2f}",
                       (x + 105, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        self.prev_frame = frame.copy()
        return display_frame

    def run(self):
        """Main detection loop"""
        if not self.initialize_camera():
            return

        # Read first frame for setup
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Cannot read from camera")
            return

        # Setup machine circles
        circles = self.setup_machine_circles(frame)
        if not circles:
            logger.error("No machines defined. Exiting.")
            return

        # Initialize machine states
        for idx, circle_roi in enumerate(circles):
            machine = MachineState(
                machine_id=idx + 1,
                is_running=False,
                last_state_change=time.time(),
                motion_history=deque(maxlen=self.motion_history_size),
                circle_roi=circle_roi
            )
            self.machines.append(machine)

        self.setup_complete = True
        logger.info("Monitoring %s washing machines...", len(self.machines))
        logger.info("Press 'q' to quit")

        # Main detection loop
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Error reading frame")
                    break

                # Process frame and update states
                display_frame = self.process_frame(frame)

                # Show the frame
                cv2.imshow('Washing Machine Monitor', display_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Shutting down...")
                    break

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Configuration
    CAMERA_SOURCE = 0  # Use 0 for webcam, or provide video file path
    BACKEND_URL = "http://localhost:8000/api/machine-state"  # Update with your backend URL
    MOTION_THRESHOLD = 0.01 # Rotational detector outputs smaller per-frame scores; tune 0.005–0.02
    STATE_CHANGE_DELAY = 10.0  # Seconds to wait before confirming state change
    USE_DRY_RUN = True

    # Create and run detector
    detector = WashingMachineDetector(
        camera_source=CAMERA_SOURCE,
        backend_url=BACKEND_URL,
        motion_threshold=MOTION_THRESHOLD,
        state_change_delay=STATE_CHANGE_DELAY,
        disable_backend=USE_DRY_RUN,
    )

    detector.run()


if __name__ == "__main__":
    main()