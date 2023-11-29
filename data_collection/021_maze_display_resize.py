import cv2
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import argparse

class VideoCaptureApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Video Capture App")
        self.args = args

        # Load and resize the 'maze.png' image to 512x512
        maze_image = Image.open("maze.png")
        maze_image = maze_image.resize((512, 512))
        maze_image = ImageTk.PhotoImage(maze_image)
        self.image_label = ttk.Label(self.root, image=maze_image)
        self.image_label.image = maze_image
        self.image_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Create a frame for buttons and video label
        self.buttons_frame = ttk.Frame(self.root)
        self.buttons_frame.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.capture_btn = ttk.Button(self.buttons_frame, text="Start Capture", command=self.start_capture)
        self.capture_btn.grid(row=0, column=0, padx=10, pady=10)

        self.stop_btn = ttk.Button(self.buttons_frame, text="Stop Capture", command=self.stop_capture, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10)

        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        self.video_source_label = ttk.Label(self.root, text="Video Source:")
        self.video_source_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.video_source_entry = ttk.Entry(self.root)
        self.video_source_entry.insert(0, 0)  # Default to webcam (change to video file path if needed)
        self.video_source_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.output_path_label = ttk.Label(self.root, text="Output Path:")
        self.output_path_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.output_path_entry = ttk.Entry(self.root)
        self.output_path_entry.insert(0, self.args.output_path)  # Set default output path from command line
        self.output_path_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.fps_label = ttk.Label(self.root, text="FPS:")
        self.fps_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.fps_entry = ttk.Entry(self.root)
        self.fps_entry.insert(0, self.args.fps)  # Set default FPS from command line
        self.fps_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        self.video_capture = None
        self.video_writer = None
        self.capture_thread = None
        self.is_capturing = False

    def start_capture(self):
        if not self.is_capturing:
            video_source = self.video_source_entry.get()
            self.video_capture = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

            if not self.video_capture.isOpened():
                print("Error: Unable to open video source.")
                return

            self.is_capturing = True
            self.capture_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # Get video properties for the video writer
            fps = float(self.fps_entry.get())
            output_path = self.output_path_entry.get()

            # Define the codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (self.args.width, self.args.height))

            self.capture_thread = threading.Thread(target=self.capture_video)
            self.capture_thread.start()

    def capture_video(self):
        while self.is_capturing:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Resize the frame to 512x512
            frame_resized = cv2.resize(frame, (512, 512))

            # Display the frame in a label widget
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
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
    parser.add_argument("--output_path", default="output.mp4", help="Output video file path")
    parser.add_argument("--fps", type=float, default=30, help="Frames per second (FPS)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = VideoCaptureApp(root, args)
    root.mainloop()
    cv2.destroyAllWindows()