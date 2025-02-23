import cv2
import jax
import jax.numpy as jnp
import jax.numpy as jp
import numpy as np


# JAX utils

def vmap2(f):
    return jax.vmap(jax.vmap(f))


def norm(v, axis=-1, keepdims=False, eps=0.0):
    return jp.sqrt((v * v).sum(axis, keepdims=keepdims).clip(eps))


def normalize(v, axis=-1, eps=1e-20):
    return v / norm(v, axis, keepdims=True, eps=eps)


def lerp(x, a, b):
    return jp.float32(a) * (1.0 - x) + jp.float32(b) * x


def cmap_e(e):
    return 1.0 - jp.stack([e, -e], -1).clip(0) @ jp.float32([[0.3, 1, 1], [1, 0.3, 1]])


def cmap_ug(u, g):
    vis = lerp(u[..., None], [0.1, 0.1, 0.3], [0.2, 0.7, 1.0])
    return lerp(g[..., None], vis, [1.17, 0.91, 0.13])


def text_overlay(img: jnp.ndarray, text: str, pos=(20, 30), color=(255, 255, 255)):
    """
    Optimized function to overlay text on a JAX image.
    It performs a single device-to-host transfer (if needed), converts the image
    to uint8 with one call to cv2.convertScaleAbs, and does color conversion only if necessary.
    """
    # Bring the JAX DeviceArray to host memory as a numpy array.
    # Using jax.device_get is explicit and can be faster than np.asarray if the array lives on GPU.
    img_np = jax.device_get(img)

    # If the image is floating point (and assumed to be in [0,1]), convert it to 8-bit.
    # cv2.convertScaleAbs is optimized in C++.
    if img_np.dtype != np.uint8:
        img_np = cv2.convertScaleAbs(img_np, alpha=255)

    # Put the overlay text on the image.
    # OpenCV's putText is very fast (written in C/C++).
    cv2.putText(img_np, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA)
    return img_np
