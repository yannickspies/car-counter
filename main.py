import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

COLORS = {
    2: (0, 0, 255),  # Car - Red
    5: (0, 255, 255),  # Bus - Yellow
}


def draw_bounding_boxes(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        class_id = int(obj.cls[0])
        color = COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def get_center_rectangle(frame, height_factor=0.5, width_factor=0.5):
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
    results = model(frame)[0]
    bboxes = results.boxes

    # Filter for cars and buses
    objects = [box for box in bboxes if int(box.cls[0]) in [2, 5]]
    return len(objects), objects


def draw_polygon(frame, polygon):
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=6)


def process_frame(frame, frame_width, frame_height, roi_enabled):
    if roi_enabled:
        roi_polygon = get_center_rectangle(frame, height_factor=0.5, width_factor=0.5)
        roi_polygon = np.clip(roi_polygon, 0, [frame_width, frame_height])

        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [roi_polygon], (255, 255, 255))
        roi_frame = cv2.bitwise_and(frame, mask)

        num_objects, objects = count_objects_in_frame(roi_frame)

        draw_polygon(frame, roi_polygon)
    else:
        num_objects, objects = count_objects_in_frame(frame)

    draw_bounding_boxes(frame, objects)

    center_x, center_y = frame_width // 2, (frame_height // 2) - 40
    cv2.putText(
        frame,
        f"Objects: {num_objects}",
        (center_x - 100, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    return frame, num_objects


def process_videos(input_dir, roi_enabled=True):
    for filename in os.listdir(input_dir):
        if filename.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(video_path)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter(
                f"processed_{filename}",
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, num_objects = process_frame(
                    frame, frame_width, frame_height, roi_enabled
                )
                out.write(processed_frame)

                print(
                    f"Processed frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} of {filename}, Objects detected: {num_objects}"
                )

            cap.release()
            out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    input_directory = "input"
    roi_flag = True
    process_videos(input_directory, roi_enabled=roi_flag)
