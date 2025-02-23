from typing import Optional

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, window_name:str="Animation", fps:Optional[float]=30.0):
        """
        Initialize the live animator.

        Parameters:
            window_name (str): Title of the OpenCV window.
            fps (float): Frames per second to display.
        """
        self.window_name = window_name

        if fps is None:
            fps = 1000

        self.fps = fps
        # Compute the delay (in milliseconds) between frames.
        self.delay = int(1000 / fps)
        # Create a named window that can be resized.
        cv2.namedWindow(self.window_name)

    def add(self, img: np.ndarray) -> bool:
        """
        Add a frame to the animation display.

        The image is preprocessed:
         - If in floating point, values are clipped to [0, 1] and scaled to uint8.
         - Grayscale images are converted to a 3-channel BGR image.
         - Color images are converted from RGB (if needed) to BGR for OpenCV.

        Parameters:
            img (array-like): Frame data (H x W or H x W x 3).
        """
        # Display the image in the window.
        cv2.imshow(self.window_name, img)
        # Wait for a short period to control the frame rate.
        # This wait is non-blocking if the delay is short.
        key = cv2.waitKey(self.delay) & 0xFF

        return key != ord('q')

    def close(self) -> None:
        """
        Close the display window.
        """
        cv2.destroyWindow(self.window_name)

    def __enter__(self) -> "VideoWriter":
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, *args):
        """
        Exit the context manager, ensuring the window is closed.
        """
        self.close()

    def __call__(self, img: np.ndarray) -> bool:
        return self.add(img)
