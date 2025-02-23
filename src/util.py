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


def text_overlay(img: np.array, text: str, pos=(20, 30), color=(255, 255, 255)):
    """
    Optimized function to overlay text on a JAX image.
    It performs a single device-to-host transfer (if needed), converts the image
    to uint8 with one call to cv2.convertScaleAbs, and does color conversion only if necessary.
    """
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=color, thickness=2, lineType=cv2.LINE_AA)
    return img


# a CPU, uint8, BGR image. For example, if your simulation originally outputs a JAX array in [0, 1] RGB,
# you can convert it once as follows:
def convert_to_image(img: jnp.ndarray):
    # Transfer from device to host (once per frame) and scale/clip.
    img_cpu = jax.device_get(img)
    if img_cpu.dtype != np.uint8:
        # Convert from float [0,1] to uint8 and clip, then convert from RGB to BGR.
        img_cpu = cv2.convertScaleAbs(img_cpu, alpha=255)

    img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)

    return img_cpu
