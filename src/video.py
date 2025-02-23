import cv2
import numpy as np

class VideoWriter:
    def __init__(self, window_name="Animation", fps=30.0):
        """
        Initialize the live animator.

        Parameters:
            window_name (str): Title of the OpenCV window.
            fps (float): Frames per second to display.
        """
        self.window_name = window_name
        self.fps = fps
        # Compute the delay (in milliseconds) between frames.
        self.delay = int(1000 / fps)
        # Create a named window that can be resized.
        cv2.namedWindow(self.window_name)

    def add(self, img) -> bool:
        """
        Add a frame to the animation display.

        The image is preprocessed:
         - If in floating point, values are clipped to [0, 1] and scaled to uint8.
         - Grayscale images are converted to a 3-channel BGR image.
         - Color images are converted from RGB (if needed) to BGR for OpenCV.

        Parameters:
            img (array-like): Frame data (H x W or H x W x 3).
        """
        # Ensure the image is a NumPy array.
        img = np.asarray(img)

        # If the image is in floating point, convert it to uint8.
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(np.clip(img, 0, 1) * 255)

        # Convert grayscale to a 3-channel image.
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            # Assuming input is in RGB order; convert to BGR for OpenCV.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Display the image in the window.
        cv2.imshow(self.window_name, img)
        # Wait for a short period to control the frame rate.
        # This wait is non-blocking if the delay is short.
        key = cv2.waitKey(self.delay)

        if key == ord('q'):
            return False
        return True

    def close(self):
        """
        Close the display window.
        """
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, *args):
        """
        Exit the context manager, ensuring the window is closed.
        """
        self.close()

    def __call__(self, img) -> bool:
        return self.add(img)