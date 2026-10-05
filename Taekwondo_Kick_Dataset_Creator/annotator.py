import cv2 as cv
import csv
import os

def annotate_video(video_path, output_csv="annotations.csv"):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    video_name = os.path.basename(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)

    intervals = []
    current_start = None

    print(f"\n--- Annotating: {video_name} ---")
    print("Controls:")
    print("  's' -> Mark START of a kick")
    print("  'e' -> Mark END of a kick")
    print("  'd' -> Next frame")
    print("  'a' -> Previous frame")
    print("  'q' -> Save and Quit")
    print("----------------------------------")

    frame_idx = 0
    while True:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Add overlay info
        display_frame = frame.copy()
        cv.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(display_frame, f"Start: {current_start if current_start is not None else 'None'}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Annotator", display_frame)
        key = cv.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord('a'):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('s'):
            current_start = frame_idx
            print(f"Start marked at: {frame_idx}")
        elif key == ord('e'):
            if current_start is not None:
                intervals.append((video_name, current_start, frame_idx))
                print(f"Interval saved: {current_start} to {frame_idx}")
                current_start = None
            else:
                print("Error: Mark the start ('s') before marking the end ('e')!")

    cap.release()
    cv.destroyAllWindows()

    # Save to CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['video', 'start_frame', 'end_frame'])
        for interval in intervals:
            writer.writerow(interval)

    print(f"\nAnnotations saved to {output_csv}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(script_dir, "videos")
    output_csv_path = os.path.join(script_dir, "annotations.csv")

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
        print(f"Created {video_folder} folder. Please add videos there.")
    else:
        while True:
            videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
            if not videos:
                print(f"No videos found in {video_folder}")
                break

            print("\n--- Video Selection ---")
            print("Available videos:")
            for i, v in enumerate(videos):
                print(f"{i}: {v}")
            print("x: Exit program")

            try:
                user_input = input("Select video index to annotate (or 'x' to exit): ").strip().lower()
                if user_input == 'x':
                    print("Exiting annotator.")
                    break

                choice = int(user_input)
                annotate_video(os.path.join(video_folder, videos[choice]), output_csv=output_csv_path)
            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")
