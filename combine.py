import cv2
import numpy as np
import argparse
from tqdm import tqdm

def resize_frame(frame, target_width, target_height):
    """Resize a frame to the target dimensions."""
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

def combine_videos(video1_path, video2_path, output_path):
    """Combine two videos by using a mask where a specific color determines which video's pixel to use."""
    # Open video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get properties of the first video
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define the key color (magenta: FF00FF in RGB) and a tolerance
    key_color = np.array([255, 0, 255])
    tolerance = 30

    # Process frames with a progress bar
    with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            # Break if either video ends
            if not ret1 or not ret2:
                break

            # Resize frame2 if necessary
            if frame2.shape[1] != width or frame2.shape[0] != height:
                frame2 = resize_frame(frame2, width, height)

            # Create a mask where frame1 is close to the key color within the tolerance
            diff = np.abs(frame1 - key_color)
            mask = np.all(diff <= tolerance, axis=-1)

            # Create the combined frame
            combined_frame = frame1.copy()
            combined_frame[mask] = frame2[mask]

            # Write the frame to the output video
            out.write(combined_frame)

            # Update progress bar
            pbar.update(1)

    # Release resources
    cap1.release()
    cap2.release()
    out.release()

    print(f"Combined video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine two videos using a mask based on a key color (FF00FF).")
    parser.add_argument("video1", help="Path to the first video")
    parser.add_argument("video2", help="Path to the second video")
    parser.add_argument("output", help="Path to save the combined video")
    args = parser.parse_args()

    combine_videos(args.video1, args.video2, args.output)

if __name__ == "__main__":
    main()
