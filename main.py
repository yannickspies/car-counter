import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Medium model for better accuracy)
model = YOLO("yolov8m.pt")

# Colors for different classes
COLORS = {
    2: (0, 255, 0),  # Car - Green
    5: (255, 0, 0),  # Bus - Blue (Class ID 5 for 'bus' in COCO dataset)
}


def draw_bounding_boxes(frame, objects):
    """Draw bounding boxes around detected objects."""
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        class_id = int(obj.cls[0])
        color = COLORS.get(
            class_id, (0, 255, 255)
        )  # Default color is yellow if class is not in COLORS
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def get_center_rectangle(frame, height_factor=0.5, width_factor=0.5):
    """Get a centered rectangle in the middle of the frame for the region of interest (ROI)."""
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2
    half_height = int((frame_height * height_factor) // 2)
    half_width = int((frame_width * width_factor) // 2)
    return np.array(
        [
            [center_x - half_width, center_y - half_height],
            [center_x + half_width, center_y - half_height],
            [center_x + half_width, center_y + half_height],
            [center_x - half_width, center_y + half_height],
        ]
    )


def count_objects_in_frame(frame):
    """Count cars and buses in the frame or ROI."""
    results = model(frame)[0]
    bboxes = results.boxes  # Get boxes from the first result
    objects = [
        box for box in bboxes if int(box.cls[0]) in [2, 5]
    ]  # Filter for cars and buses
    return len(objects), objects


def draw_polygon(frame, polygon):
    """Draw the region of interest (ROI) polygon on the frame."""
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=6)


def process_frame(frame, frame_width, frame_height):
    """Process each frame to detect objects and draw bounding boxes and ROI."""
    # Get the centered rectangle covering 50% of the frame
    roi_polygon = get_center_rectangle(frame, height_factor=0.5, width_factor=0.5)
    roi_polygon = np.clip(roi_polygon, 0, [frame_width, frame_height])

    # Create a mask with the polygon
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_polygon], (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)

    # Count cars and buses in the ROI
    num_objects, objects = count_objects_in_frame(roi_frame)

    # Draw the polygon on the original frame
    draw_polygon(frame, roi_polygon)

    # Draw bounding boxes around detected objects
    draw_bounding_boxes(frame, objects)

    # Display the object count centered and large above the ROI in red
    center_x, center_y = frame_width // 2, roi_polygon[0][1] - 40
    cv2.putText(
        frame,
        f"Objects: {num_objects}",
        (center_x - 100, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )

    return frame, num_objects


def main():
    # Open video capture (use 0 for webcam or path to video file)
    cap = cv2.VideoCapture("cars-driving-on-highway.mp4")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(
        "output_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, num_objects = process_frame(frame, frame_width, frame_height)
        out.write(processed_frame)

        # Log useful information to the console
        print(
            f"Processed frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, Objects detected: {num_objects}"
        )

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
