import cv2
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import argparse

class VideoCaptureApp:
    def __init__(self, root, output_path, fps):
        self.root = root
        self.root.title("Video Capture App")

        self.capture_btn = ttk.Button(self.root, text="Start Capture", command=self.start_capture)
        self.capture_btn.pack()

        self.stop_btn = ttk.Button(self.root, text="Stop Capture", command=self.stop_capture, state="disabled")
        self.stop_btn.pack()

        self.save_btn = ttk.Button(self.root, text="Save Video", command=self.save_video, state="disabled")
        self.save_btn.pack()

        self.video_source_label = ttk.Label(self.root, text="Video Source:")
        self.video_source_label.pack()

        self.video_source_entry = ttk.Entry(self.root)
        self.video_source_entry.insert(0, 0)  # Default to webcam (change to video file path if needed)
        self.video_source_entry.pack()

        self.output_path_label = ttk.Label(self.root, text="Output Path:")
        self.output_path_label.pack()

        self.output_path_entry = ttk.Entry(self.root)
        self.output_path_entry.insert(0, output_path)  # Set default output path from command line
        self.output_path_entry.pack()

        self.fps_label = ttk.Label(self.root, text="FPS:")
        self.fps_label.pack()

        self.fps_entry = ttk.Entry(self.root)
        self.fps_entry.insert(0, fps)  # Set default FPS from command line
        self.fps_entry.pack()

        self.video_capture = None
        self.video_writer = None
        self.capture_thread = None
        self.is_capturing = False

        self.video_label = ttk.Label(self.root)
        self.video_label.pack()

    def start_capture(self):
        if not self.is_capturing:
            video_source = self.video_source_entry.get()
            self.video_capture = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)

            if not self.video_capture.isOpened():
                print("Error: Unable to open video source.")
                return

            self.is_capturing = True
            self.capture_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # Get video properties for the video writer
            frame_width = int(self.video_capture.get(3))
            frame_height = int(self.video_capture.get(4))
            fps = float(self.fps_entry.get())  # Adjust this as needed

            output_path = self.output_path_entry.get()

            # Define the codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            self.capture_thread = threading.Thread(target=self.capture_video)
            self.capture_thread.start()

    def capture_video(self):
        while self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Display the frame in a label widget
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep a reference to prevent garbage collection

            # Write the frame to the video file
            self.video_writer.write(frame)

        self.video_capture.release()
        self.is_capturing = False
        self.capture_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        # Automatically save the video when capturing stops
        self.save_video()

    def stop_capture(self):
        self.is_capturing = False

    def save_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print("Video saved as " + self.output_path_entry.get())

def parse_args():
    parser = argparse.ArgumentParser(description="Video Capture App")
    parser.add_argument("--output", default="output.mp4", help="Output video file path")
    parser.add_argument("--fps", type=float, default=30, help="Frames per second (FPS)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = VideoCaptureApp(root, args.output, args.fps)
    root.mainloop()
    cv2.destroyAllWindows()
