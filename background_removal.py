"""
Real-Time Background Removal Script for Streaming and Virtual Cameras

This script uses MediaPipe's Selfie Segmentation model to perform real-time
background removal. It replaces the background with a solid green screen,
which can be chroma-keyed in tools like OBS or Zoom. Features include
temporal smoothing and mask refinement using Gaussian blur and morphological
erosion.

Author: Pi Ko
Email: pi.ko@nyu.edu
"""

import cv2
import mediapipe as mp
import numpy as np


def list_connected_cameras():
    """
    Lists all connected cameras by attempting to open camera indexes sequentially.

    Returns:
        list[int]: A list of valid camera indexes.
    """
    index = 0
    valid_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                valid_cameras.append(index)
            cap.release()
        else:
            break
        index += 1
    return valid_cameras


def main():
    """
    Main function for real-time background removal. It allows the user to:
    - Select a camera index.
    - Perform segmentation on the video feed.
    - Replace the background with a solid green screen.
    - View the processed output in real-time.
    """
    # Step 1: Detect available cameras
    cameras = list_connected_cameras()
    if not cameras:
        print("No cameras found!")
        return

    print("Available camera indexes:", cameras)

    # Step 2: Prompt the user to select a camera
    cam_index = None
    while cam_index not in cameras:
        try:
            cam_index = int(input(f"Select a camera index from the above list: "))
        except ValueError:
            print("Invalid input. Please enter a valid camera index.")

    # Step 3: Open the selected camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Failed to open camera with index {cam_index}")
        return

    # Step 4: Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Background color (green screen)
    bg_color = (0, 255, 0)

    # Temporal smoothing setup
    prev_blurred_mask = None  # Store the previous mask for smoothing
    alpha = 0.6  # Blending factor for smoothing

    # Morphological erosion parameters
    erosion_kernel_size = 5  # Kernel size for erosion
    erosion_iterations = 1  # Number of erosion iterations
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))

    print("Press 'q' to exit.")

    while True:
        # Step 5: Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting...")
            break

        # Step 6: Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 7: Perform segmentation to obtain the mask
        results = selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask  # Values range from 0.0 to 1.0

        # Step 8: Smooth the mask with Gaussian blur
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Step 9: Apply temporal smoothing
        if prev_blurred_mask is not None:
            blurred_mask = alpha * prev_blurred_mask + (1 - alpha) * blurred_mask
        prev_blurred_mask = blurred_mask

        # Step 10: Threshold the mask to create a binary condition
        threshold_value = 0.5
        condition = blurred_mask > threshold_value

        # Step 11: Refine the mask using morphological erosion
        mask_uint8 = (condition.astype(np.uint8)) * 255  # Convert to 0-255
        eroded_mask = cv2.erode(mask_uint8, erosion_kernel, iterations=erosion_iterations)
        final_condition = eroded_mask > 128  # Convert back to a boolean condition

        # Step 12: Create a green background
        bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        bg_frame[:] = bg_color

        # Step 13: Blend the original frame and the green background
        output_frame = np.where(final_condition[..., None], frame, bg_frame)

        # Step 14: Display the output frame
        cv2.imshow("Green Background - Eroded Mask", output_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
