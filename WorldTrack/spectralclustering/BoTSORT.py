import os
import glob
import re
from pathlib import Path
from ultralytics import YOLO
import cv2

# Hard-coded input folder path
INPUT_FOLDER = "path/to/single/cam/images"
OUTPUT_FILE = "BoTSORT_results/mmcows_test_cam1.txt"
OUTPUT_VIS_FOLDER = "BoTSORT_results/visualizations_mmcows_test_cam1"  # Change this to any folder name you want

# Create output visualization folder if it doesn't exist
os.makedirs(OUTPUT_VIS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load the YOLO model
model = YOLO(
    "/path/to/trained/YOLOmodel/weights/best.pt")  # Replace with your model path (e.g., "yolo11n.pt")

# Get all image files from the input folder (ignoring JSON files)
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))


# Function to extract numeric part from filename for sorting
def extract_number(filepath):
    filename = os.path.basename(filepath)
    # Extract all digits from the filename
    numbers = re.findall(r'\d+', filename)
    # Return the first number found (assuming it's the frame number)
    return int(numbers[0]) if numbers else 0


# Sort images numerically by filename
image_files = sorted(image_files, key=extract_number)

if not image_files:
    print("No images found in the input folder!")
    exit()

print(f"Found {len(image_files)} images to process")

# Open output file for writing MOT format results
with open(OUTPUT_FILE, 'w') as f:
    # Process each image
    for frame_idx, image_path in enumerate(image_files, start=1):
        print(f"Processing frame {frame_idx}/{len(image_files)}: {os.path.basename(image_path)}")

        # Read the image for visualization
        img = cv2.imread(image_path)

        # Run YOLO tracking on the image with BoTSORT
        results = model.track(image_path, persist=True, tracker="botsort.yaml")

        # Extract tracking results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # Track IDs
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

            # Convert to MOT format and write to file
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Convert from xyxy to xywh (top-left corner + width + height)
                bb_left = box[0]
                bb_top = box[1]
                bb_width = box[2] - box[0]
                bb_height = box[3] - box[1]

                # Write in MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                line = f"{frame_idx},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},{conf},-1,-1,-1\n"
                f.write(line)

                # Draw bounding box on the image
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Draw rectangle (bounding box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prepare label text with ID and confidence
                label = f"ID:{track_id} {conf:.2f}"

                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Draw background rectangle for text
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)

                # Draw text (ID and confidence)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Save the annotated image
        output_image_name = f"frame_{frame_idx:06d}.jpg"  # You can also use os.path.basename(image_path) to keep original name
        output_image_path = os.path.join(OUTPUT_VIS_FOLDER, output_image_name)
        cv2.imwrite(output_image_path, img)

print(f"Tracking complete! Results saved to {OUTPUT_FILE}")
print(f"Visualizations saved to {OUTPUT_VIS_FOLDER}")