"""
QuAdGate - Quadrilateral Gate Corner Detection

Detects the 4 corners of racing gates from segmentation masks.
Uses contour detection and quadrilateral fitting with robust
handling of partial visibility and occlusion.
"""

import numpy as np
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import cv2


@dataclass
class GateDetection:
    """Result of gate corner detection."""
    corners: np.ndarray  # 4x2 corner coordinates (TL, TR, BR, BL order)
    confidence: float  # Detection confidence [0, 1]
    is_complete: bool  # True if all 4 corners detected
    visible_corners: int  # Number of visible corners (0-4)
    contour: Optional[np.ndarray] = None  # Original contour
    center: Optional[np.ndarray] = None  # Gate center in image


class QuAdGate:
    """
    Quadrilateral gate corner detector.

    Processes segmentation masks to extract gate corner positions.
    Handles partial visibility by fitting quadrilaterals to contours.
    """

    def __init__(
        self,
        min_contour_area: int = 100,
        epsilon_factor: float = 0.02,
        max_corner_distance: float = 10.0,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize QuAdGate.

        Args:
            min_contour_area: Minimum contour area in pixels to consider
            epsilon_factor: Factor for contour approximation (relative to perimeter)
            max_corner_distance: Maximum distance to match corners between frames
            confidence_threshold: Minimum confidence to report detection
        """
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
        self.max_corner_distance = max_corner_distance
        self.confidence_threshold = confidence_threshold

        # Expected corner order (top-left, top-right, bottom-right, bottom-left)
        self._corner_order = ["TL", "TR", "BR", "BL"]

    def detect(
        self,
        mask: np.ndarray,
        return_all: bool = False,
    ) -> Optional[GateDetection]:
        """
        Detect gate corners from segmentation mask.

        Args:
            mask: Binary segmentation mask (H, W), values in [0, 1] or [0, 255]
            return_all: If True, return all detected gates (not just best)

        Returns:
            GateDetection object or None if no gate detected
        """
        # Ensure mask is uint8
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).astype(np.uint8)

        # Threshold if needed
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Process each contour
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            detection = self._process_contour(contour, mask.shape) # type: ignore
            if detection is not None:
                detections.append(detection)

        if not detections:
            return None

        # Return best detection (highest confidence)
        # if return_all:
        #     return detections

        return max(detections, key=lambda d: d.confidence)

    def _process_contour(
        self,
        contour: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[GateDetection]:
        """
        Process a single contour to extract gate corners.

        Args:
            contour: OpenCV contour
            image_shape: (H, W) shape of source image

        Returns:
            GateDetection or None
        """
        # Compute contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return None

        # Approximate contour to polygon
        epsilon = self.epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get corner count
        num_vertices = len(approx)

        # Compute center
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None

        center = np.array([
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ])

        # Try to fit quadrilateral
        if num_vertices == 4:
            # Perfect case: exactly 4 corners
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = self._order_corners(corners)
            confidence = self._compute_confidence(corners, area, perimeter)
            return GateDetection(
                corners=corners,
                confidence=confidence,
                is_complete=True,
                visible_corners=4,
                contour=contour,
                center=center,
            )

        elif num_vertices == 3:
            # Triangle: estimate 4th corner
            corners = approx.reshape(3, 2).astype(np.float32)
            corners = self._complete_from_triangle(corners, center)
            confidence = self._compute_confidence(corners, area, perimeter) * 0.7
            return GateDetection(
                corners=corners,
                confidence=confidence,
                is_complete=False,
                visible_corners=3,
                contour=contour,
                center=center,
            )

        elif num_vertices > 4:
            # Too many vertices: fit minimum area rectangle
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
            corners = self._order_corners(corners)
            # Lower confidence for fitted rectangle
            confidence = self._compute_confidence(corners, area, perimeter) * 0.8
            return GateDetection(
                corners=corners,
                confidence=confidence,
                is_complete=True,
                visible_corners=4,
                contour=contour,
                center=center,
            )

        else:
            # Too few vertices: fit minimum area rectangle
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
            corners = self._order_corners(corners)
            confidence = self._compute_confidence(corners, area, perimeter) * 0.5
            return GateDetection(
                corners=corners,
                confidence=confidence,
                is_complete=False,
                visible_corners=num_vertices,
                contour=contour,
                center=center,
            )

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.

        Args:
            corners: 4x2 array of corner coordinates

        Returns:
            Ordered 4x2 array
        """
        # Sort by sum of coordinates (top-left has smallest sum)
        sum_coords = corners.sum(axis=1)
        tl_idx = np.argmin(sum_coords)
        br_idx = np.argmax(sum_coords)

        # Sort by difference (top-right has smallest difference)
        diff_coords = np.diff(corners, axis=1).flatten()
        tr_idx = np.argmin(diff_coords)
        bl_idx = np.argmax(diff_coords)

        # Handle potential conflicts
        indices = [tl_idx, tr_idx, br_idx, bl_idx]
        if len(set(indices)) != 4:
            # Fallback: use centroid-based ordering
            center = corners.mean(axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            order = np.argsort(angles)
            # Rotate so that top-left is first
            start_idx = np.argmin(corners[order].sum(axis=1))
            order = np.roll(order, -start_idx)
            return corners[order]

        return corners[[tl_idx, tr_idx, br_idx, bl_idx]]

    def _complete_from_triangle(
        self,
        corners: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate 4th corner from 3 visible corners.

        Uses parallelogram assumption: opposite corners sum to same point.

        Args:
            corners: 3x2 array of visible corners
            center: Gate center estimate

        Returns:
            4x2 array with estimated 4th corner
        """
        # Find which corner is missing based on expected rectangle properties
        # Estimate 4th corner using parallelogram rule
        # P4 = P1 + P3 - P2 (for opposite corners)

        # Try each combination
        best_fourth = None
        best_score = float("inf")

        for i in range(3):
            p1, p2, p3 = corners[i], corners[(i+1) % 3], corners[(i+2) % 3]
            # P4 = P1 + P3 - P2
            p4 = p1 + p3 - p2

            # Score based on how rectangular the result is
            all_corners = np.vstack([corners, p4.reshape(1, 2)])
            score = self._rectangularity_score(all_corners)

            if score < best_score:
                best_score = score
                best_fourth = p4

        if best_fourth is None:
            # Fallback: mirror point across center
            farthest_idx = np.argmax(np.linalg.norm(corners - center, axis=1))
            best_fourth = 2 * center - corners[farthest_idx]

        all_corners = np.vstack([corners, best_fourth.reshape(1, 2)])
        return self._order_corners(all_corners)

    def _rectangularity_score(self, corners: np.ndarray) -> float:
        """
        Compute how rectangular a quadrilateral is.

        Lower score = more rectangular.

        Args:
            corners: 4x2 array of corners

        Returns:
            Rectangularity deviation score
        """
        # Check if opposite sides are parallel and equal length
        ordered = self._order_corners(corners)

        # Side vectors
        sides = np.array([
            ordered[1] - ordered[0],  # Top
            ordered[2] - ordered[1],  # Right
            ordered[3] - ordered[2],  # Bottom
            ordered[0] - ordered[3],  # Left
        ])

        # Opposite sides should be parallel (dot product of perpendiculars = 0)
        # and equal length
        length_diff = abs(np.linalg.norm(sides[0]) - np.linalg.norm(sides[2]))
        length_diff += abs(np.linalg.norm(sides[1]) - np.linalg.norm(sides[3]))

        # Check angles (should be 90 degrees)
        angle_dev = 0.0
        for i in range(4):
            v1 = sides[i]
            v2 = sides[(i+1) % 4]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle_dev += abs(cos_angle)  # Should be 0 for 90 degrees

        return length_diff + angle_dev # type: ignore

    def _compute_confidence(
        self,
        corners: np.ndarray,
        area: float,
        perimeter: float,
    ) -> float:
        """
        Compute detection confidence.

        Based on:
        - Rectangularity of detected shape
        - Area relative to expected
        - Aspect ratio plausibility

        Args:
            corners: 4x2 corner array
            area: Contour area
            perimeter: Contour perimeter

        Returns:
            Confidence score [0, 1]
        """
        # Rectangularity score (lower is better)
        rect_score = self._rectangularity_score(corners)
        rect_confidence = max(0, 1 - rect_score / 10)

        # Area score (penalize very small or very large)
        # Expected area range based on gate at various distances
        min_expected_area = 500
        max_expected_area = 80000  # Most of frame
        if area < min_expected_area:
            area_confidence = area / min_expected_area
        elif area > max_expected_area:
            area_confidence = max_expected_area / area
        else:
            area_confidence = 1.0

        # Aspect ratio score (gates are roughly square)
        ordered = self._order_corners(corners)
        width = np.linalg.norm(ordered[1] - ordered[0])
        height = np.linalg.norm(ordered[3] - ordered[0])
        aspect_ratio = width / (height + 1e-6)

        # Expected aspect ratio range [0.5, 2.0]
        if 0.5 <= aspect_ratio <= 2.0:
            aspect_confidence = 1.0
        else:
            aspect_confidence = 0.5

        # Combine scores
        confidence = (rect_confidence * 0.4 + area_confidence * 0.3 + aspect_confidence * 0.3)
        return float(np.clip(confidence, 0, 1))

    def track_corners(
        self,
        prev_detection: GateDetection,
        curr_detection: GateDetection,
    ) -> GateDetection:
        """
        Track corners between frames for temporal consistency.

        Args:
            prev_detection: Previous frame detection
            curr_detection: Current frame detection

        Returns:
            Updated current detection with tracked corners
        """
        if prev_detection is None:
            return curr_detection

        # Match corners based on distance
        prev_corners = prev_detection.corners
        curr_corners = curr_detection.corners

        # Compute distance matrix
        distances = np.linalg.norm(
            prev_corners[:, np.newaxis, :] - curr_corners[np.newaxis, :, :],
            axis=2
        )

        # Find best matching (Hungarian algorithm simplified)
        matched_curr = curr_corners.copy()
        used = set()

        for i in range(4):
            min_dist = float("inf")
            min_j = -1

            for j in range(4):
                if j not in used and distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    min_j = j

            if min_j >= 0 and min_dist < self.max_corner_distance:
                # Smooth corner position
                alpha = 0.7  # Current frame weight
                matched_curr[i] = alpha * curr_corners[min_j] + (1 - alpha) * prev_corners[i]
                used.add(min_j)
            else:
                # Keep previous position if no good match
                matched_curr[i] = prev_corners[i]

        return GateDetection(
            corners=matched_curr,
            confidence=curr_detection.confidence,
            is_complete=curr_detection.is_complete,
            visible_corners=curr_detection.visible_corners,
            contour=curr_detection.contour,
            center=curr_detection.center,
        )


class GateTracker:
    """
    Multi-gate tracker with temporal filtering.

    Tracks multiple gates across frames and handles
    gate switching during racing.
    """

    def __init__(
        self,
        detector: QuAdGate,
        smoothing_alpha: float = 0.7,
        max_missed_frames: int = 5,
    ):
        """
        Initialize tracker.

        Args:
            detector: QuAdGate detector instance
            smoothing_alpha: Exponential smoothing factor
            max_missed_frames: Max frames before dropping track
        """
        self.detector = detector
        self.smoothing_alpha = smoothing_alpha
        self.max_missed_frames = max_missed_frames

        # Tracking state
        self.current_detection: Optional[GateDetection] = None
        self.missed_frames = 0
        self.frame_count = 0

    def update(self, mask: np.ndarray) -> Optional[GateDetection]:
        """
        Process new frame and update tracking.

        Args:
            mask: Segmentation mask

        Returns:
            Current gate detection or None
        """
        self.frame_count += 1

        # Detect gates in current frame
        detection = self.detector.detect(mask)

        if detection is not None:
            # Apply temporal smoothing
            if self.current_detection is not None:
                detection = self.detector.track_corners(
                    self.current_detection, detection
                )

            self.current_detection = detection
            self.missed_frames = 0
        else:
            self.missed_frames += 1

            # Drop track if too many missed frames
            if self.missed_frames > self.max_missed_frames:
                self.current_detection = None

        return self.current_detection

    def reset(self):
        """Reset tracker state."""
        self.current_detection = None
        self.missed_frames = 0
        self.frame_count = 0


def visualize_detection(
    image: np.ndarray,
    detection: Optional[GateDetection],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Visualize gate detection on image.

    Args:
        image: RGB image (H, W, 3)
        detection: Gate detection result
        mask: Optional segmentation mask to overlay

    Returns:
        Annotated image
    """
    vis = image.copy()
    if vis.dtype != np.uint8:
        vis = (vis * 255).astype(np.uint8)

    # Overlay mask if provided
    if mask is not None:
        mask_vis = np.zeros_like(vis)
        mask_vis[:, :, 1] = (mask * 100).astype(np.uint8)  # Green overlay
        vis = cv2.addWeighted(vis, 0.7, mask_vis, 0.3, 0)

    if detection is None:
        return vis

    # Draw corners
    colors = [
        (255, 0, 0),    # TL: Red
        (0, 255, 0),    # TR: Green
        (0, 0, 255),    # BR: Blue
        (255, 255, 0),  # BL: Yellow
    ]

    corners = detection.corners.astype(np.int32)

    for i, (corner, color) in enumerate(zip(corners, colors)):
        cv2.circle(vis, tuple(corner), 3, color, -1)
        cv2.putText(vis, str(i), tuple(corner + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1)

    # Draw quadrilateral
    cv2.polylines(vis, [corners], True, (0, 255, 255), 1)

    # Draw center
    if detection.center is not None:
        center = detection.center.astype(np.int32)
        cv2.circle(vis, tuple(center), 2, (255, 0, 255), -1)

    # Add confidence text
    cv2.putText(
        vis,
        f"Conf: {detection.confidence:.2f}",
        (5, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )

    return vis


def visualize_mask_with_detection(
    mask: np.ndarray,
    detection: Optional[GateDetection],
) -> np.ndarray:
    """
    Visualize gate detection on binary mask image.

    Draws:
    - Contour in blue
    - Detected corners in different colors
    - Quadrilateral fitting in green
    - Center point in magenta

    Args:
        mask: Binary segmentation mask (H, W), values in [0, 1] or [0, 255]
        detection: Gate detection result

    Returns:
        Annotated RGB image with mask and detection visualization
    """
    # Convert mask to RGB for visualization
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.copy()
    
    # Create RGB image from mask
    vis = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)

    if detection is None:
        return vis

    # Draw contour if available
    if detection.contour is not None:
        cv2.drawContours(vis, [detection.contour], -1, (255, 0, 0), 2)

    # Draw corners with colors
    colors = [
        (255, 0, 0),    # TL: Red
        (0, 255, 0),    # TR: Green
        (0, 0, 255),    # BR: Blue
        (255, 255, 0),  # BL: Yellow
    ]
    corner_names = ['TL', 'TR', 'BR', 'BL']

    corners = detection.corners.astype(np.int32)

    for i, (corner, color, name) in enumerate(zip(corners, colors, corner_names)):
        # Draw corner dot
        cv2.circle(vis, tuple(corner), 4, color, -1)
        # Draw corner label
        cv2.putText(vis, name, tuple(corner + np.array([6, -2])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw quadrilateral (fitted gate)
    cv2.polylines(vis, [corners], True, (0, 255, 255), 2)

    # Draw center
    if detection.center is not None:
        center = detection.center.astype(np.int32)
        cv2.circle(vis, tuple(center), 3, (255, 0, 255), -1)
        cv2.putText(vis, 'Center', tuple(center + np.array([5, 15])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

    # Add confidence and info text
    info_text = f"Conf: {detection.confidence:.2f} | Corners: {detection.visible_corners}/4"
    cv2.putText(
        vis,
        info_text,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return vis


def visualize_image_with_detection(
    raw_image: np.ndarray,
    mask: np.ndarray,
    detection: Optional[GateDetection],
) -> np.ndarray:
    """
    Visualize gate detection on image.

    Draws:
    - Contour in blue
    - Detected corners in different colors
    - Quadrilateral fitting in green
    - Center point in magenta

    Args:
        raw_image: RGB image (H, W, 3), values in [0, 1] or [0, 255]
        mask: Binary segmentation mask (H, W), values in [0, 1] or [0, 255]
        detection: Gate detection result

    Returns:
        Annotated RGB image with detection visualization
    """
    vis = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

    # Draw semi-transparent mask first (underneath other annotations)
    if mask is not None:
        # Normalize mask to [0, 1]
        mask_normalized = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
        
        # Resize mask to match image dimensions if needed (YOLO may output different size)
        if mask_normalized.shape[:2] != vis.shape[:2]:
            vis = cv2.resize(vis, (mask_normalized.shape[1], mask_normalized.shape[0]))
        
        # Create colored mask overlay (green color)
        alpha = 0.3
        overlay_inds = mask_normalized > 0.5
        roi_vis = vis[overlay_inds]
        blended = ( (1-alpha) * roi_vis + alpha * np.array([0,255,0], dtype=np.uint8) ).astype(np.uint8)
        if blended is not None:
            vis[overlay_inds] = blended

    if detection is None:
        return vis

    # Draw contour if available
    if detection.contour is not None:
        cv2.drawContours(vis, [detection.contour], -1, (255, 191, 0), 2) # deepskyblue

    # Draw corners with colors
    color = (0, 165, 255) # orange
    corner_names = ['TL', 'TR', 'BR', 'BL']

    corners = detection.corners.astype(np.int32)

    for i, (corner, name) in enumerate(zip(corners, corner_names)):
        # Draw corner dot
        cv2.circle(vis, tuple(corner), 4, color, -1)
        # Draw corner label
        cv2.putText(vis, name, tuple(corner + np.array([6, -2])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw quadrilateral (fitted gate)
    cv2.polylines(vis, [corners], True, (0, 255, 255), 2)

    # Draw center
    if detection.center is not None:
        center = detection.center.astype(np.int32)
        cv2.circle(vis, tuple(center), 3, (255, 0, 255), -1)
        cv2.putText(vis, 'Center', tuple(center + np.array([5, 15])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

    # Add confidence and info text
    info_text = f"Conf: {detection.confidence:.2f} | Corners: {detection.visible_corners}/4"
    cv2.putText(
        vis,
        info_text,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return vis


if __name__ == "__main__":
    # Test QuAdGate detection
    print("Testing QuAdGate corner detection...")

    detector = QuAdGate()

    # Create synthetic gate mask
    mask = np.zeros((48, 64), dtype=np.uint8)

    # Draw a quadrilateral (simulating gate projection)
    corners = np.array([
        [20, 10],  # TL
        [50, 12],  # TR
        [48, 38],  # BR
        [18, 35],  # BL
    ], dtype=np.int32)

    cv2.fillPoly(mask, [corners], 255)

    # Detect
    detection = detector.detect(mask)

    if detection is not None:
        print(f"Detection confidence: {detection.confidence:.3f}")
        print(f"Complete: {detection.is_complete}")
        print(f"Visible corners: {detection.visible_corners}")
        print(f"Detected corners:\n{detection.corners}")
        print(f"Original corners:\n{corners}")

        # Compute corner error
        error = np.mean(np.abs(detection.corners - corners))
        print(f"Mean corner error: {error:.2f} pixels")
    else:
        print("No detection!")

    # Test with partial visibility (triangle)
    print("\nTesting partial visibility (3 corners)...")
    mask2 = np.zeros((48, 64), dtype=np.uint8)
    triangle = np.array([[20, 10], [50, 12], [48, 38]], dtype=np.int32)
    cv2.fillPoly(mask2, [triangle], 255)

    detection2 = detector.detect(mask2)
    if detection2 is not None:
        print(f"Partial detection confidence: {detection2.confidence:.3f}")
        print(f"Visible corners: {detection2.visible_corners}")
        print(f"Estimated corners:\n{detection2.corners}")

    print("\nTest complete!")
