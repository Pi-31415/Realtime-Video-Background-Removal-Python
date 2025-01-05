import cv2
import mediapipe as mp
import numpy as np

def list_connected_cameras():
    """
    Attempts to open camera indexes starting from 0, incrementing by 1,
    until it fails to read a frame. Returns a list of valid camera indexes.
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
    # 1. Get list of connected cameras
    cameras = list_connected_cameras()
    if not cameras:
        print("No cameras found!")
        return

    print("Available camera indexes:", cameras)

    # Prompt user to select a camera index
    cam_index = None
    while cam_index not in cameras:
        try:
            cam_index = int(input(f"Select a camera index from the above list: "))
        except ValueError:
            pass

    # Open the chosen camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Failed to open camera with index {cam_index}")
        return

    # 2. Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Background color (solid green)
    bg_color = (0, 255, 0)

    # Keep track of the previous mask for temporal smoothing
    prev_blurred_mask = None
    alpha = 0.6  # blend factor for temporal smoothing (0.6 to 0.9, typically)

    # Parameters for morphological erosion (currently unused, but left here for reference)
    erosion_kernel_size = 5
    erosion_iterations = 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (erosion_kernel_size, erosion_kernel_size))

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting...")
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation
        results = selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask  # Values range from 0.0 to 1.0

        # 1) Smooth the raw mask
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # 2) Temporal smoothing with previous frame
        if prev_blurred_mask is not None:
            blurred_mask = alpha * prev_blurred_mask + (1 - alpha) * blurred_mask
        prev_blurred_mask = blurred_mask

        # Suppose you want anything below 0.4 to become 0,
        # and anything 0.4 or above to be spread across [0..1].
        threshold_value = 0.1

        # Shift and rescale the blurred_mask so that 0.4 -> 0.0 and 1.0 -> 1.0
        temp_mask = (blurred_mask - threshold_value) / (1.0 - threshold_value)

        # 3) Use blurred_mask (0..1) directly as the alpha channel to feather edges
        alpha_mask = np.clip(temp_mask, 0, 1)  # ensure it's between 0 and 1
        alpha_mask_3d = alpha_mask[..., None]     # make it (H, W, 1) for broadcasting

        # Create a solid green background
        bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        bg_frame[:] = bg_color

        # 4) Blend foreground (frame) and background (bg_frame) using alpha
        #    Foreground weight = alpha_mask
        #    Background weight = (1 - alpha_mask)
        output_frame = (
            alpha_mask_3d * frame.astype(np.float32) +
            (1 - alpha_mask_3d) * bg_frame.astype(np.float32)
        ).astype(np.uint8)

        # Show the result
        cv2.imshow("Feathered Background Replacement", output_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
