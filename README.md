# Video Background Removal for OBS and Streaming

This Python script enables real-time video background removal using MediaPipe's Selfie Segmentation, creating a virtual green screen effect. It's designed for streaming applications like OBS, Zoom, and others that support chroma keying and virtual camera setups.

## Features
- **Real-time Background Removal**: Processes video feed in real-time to separate the subject from the background.
- **Green-Screen Effect**: Outputs a green background that can be replaced with a custom background using chroma keying in OBS or similar tools.
- **Temporal Smoothing**: Reduces flickering in the segmentation mask by blending it with previous frames.
- **Mask Refinement**: Applies Gaussian blur and morphological erosion for smoother and more accurate segmentation.

## Requirements
- **Python**: Version 3.8 or later.
- **Libraries**:
  - OpenCV
  - MediaPipe
  - NumPy

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/video-background-removal.git
cd video-background-removal
```

### 2. Install Dependencies
Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Run the Script
Launch the Python script:

```bash
python background_removal.py
```

## Usage
1. **Select Camera**: The script will list all available camera indexes. Choose the desired camera index.
2. **Run Background Removal**: The video feed will display the subject with a green-screen background.
3. **Integration with OBS/Streaming Tools**:
   - Configure OBS or your preferred tool to use the camera feed.
   - Apply a chroma key filter to replace the green background with your desired image or video.
4. **Exit**: Press `q` to quit the script.

## Key Parameters
- **`bg_color`**: Default green background color `(0, 255, 0)`.
- **Temporal Smoothing**: Adjust the `alpha` value (default `0.6`) to control the blend factor for mask smoothing.
- **Morphological Erosion**:
  - **Kernel Size**: Default `5` (increase for more aggressive erosion).
  - **Iterations**: Default `1` (increase to shrink the subject area more).

## Troubleshooting
- **No Camera Found**: Ensure your webcam or external camera is properly connected and detected by the operating system.
- **High CPU Usage**: Reduce video resolution or frame rate for better performance.
- **Mask Artifacts**: Fine-tune the smoothing and erosion parameters for better results.

## Example Output
The processed video will show the subject with a solid green background:

