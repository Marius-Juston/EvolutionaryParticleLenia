import PIL
import PIL.ImageDraw
import PIL.ImageFont
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


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


fontpath = plt.matplotlib.get_data_path() + '/fonts/ttf/DejaVuSansMono.ttf'
pil_font = PIL.ImageFont.truetype(fontpath, size=16)

def text_overlay(img, text, pos=(20, 10), color=(255, 255, 255)):
    img = np2pil(img)
    draw = PIL.ImageDraw.Draw(img)
    draw.text(pos, text, fill=color, font=pil_font)
    return img
