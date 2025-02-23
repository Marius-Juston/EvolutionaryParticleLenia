from collections import namedtuple
from functools import partial
from typing import Optional

import PIL
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image

plt.rcParams.update({"axes.grid": True})


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


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


# JAX utils

def vmap2(f):
    return jax.vmap(jax.vmap(f))


def norm(v, axis=-1, keepdims=False, eps=0.0):
    return jp.sqrt((v * v).sum(axis, keepdims=keepdims).clip(eps))


def normalize(v, axis=-1, eps=1e-20):
    return v / norm(v, axis, keepdims=True, eps=eps)


Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')


def peak_f(x, mu, sigma):
    return jp.exp(-((x - mu) / sigma) ** 2)


def fields_f(p: Params, points, x):
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
    U = peak_f(r, p.mu_k, p.sigma_k).sum() * p.w_k
    G = peak_f(U, p.mu_g, p.sigma_g)
    R = p.c_rep / 2 * ((1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)


def motion_f(params, points):
    grad_E = jax.grad(lambda x: fields_f(params, points, x).E)
    return -jax.vmap(grad_E)(points)


def lerp(x, a, b):
    return jp.float32(a) * (1.0 - x) + jp.float32(b) * x


def cmap_e(e):
    return 1.0 - jp.stack([e, -e], -1).clip(0) @ jp.float32([[0.3, 1, 1], [1, 0.3, 1]])


def cmap_ug(u, g):
    vis = lerp(u[..., None], [0.1, 0.1, 0.3], [0.2, 0.7, 1.0])
    return lerp(g[..., None], vis, [1.17, 0.91, 0.13])


@partial(jax.jit, static_argnames=['w', 'show_UG', 'show_cmap'])
def show_lenia(params, points, extent, w=400, show_UG=False, show_cmap=True):
    xy = jp.mgrid[-1:1:w * 1j, -1:1:w * 1j].T * extent
    e0 = -peak_f(0.0, params.mu_g, params.sigma_g)
    f = partial(fields_f, params, points)
    fields = vmap2(f)(xy)
    r2 = jp.square(xy[..., None, :] - points).sum(-1).min(-1)
    points_mask = (r2 / 0.02).clip(0, 1.0)[..., None]
    vis = cmap_e(fields.E - e0) * points_mask
    if show_cmap:
        e_mean = jax.vmap(f)(points).E.mean()
        bar = np.r_[0.5:-0.5:w * 1j]
        bar = cmap_e(bar) * (1.0 - peak_f(bar, e_mean - e0, 0.005)[:, None])
        vis = jp.hstack([vis, bar[:, None].repeat(16, 1)])
    if show_UG:
        vis_u = cmap_ug(fields.U, fields.G) * points_mask
        if show_cmap:
            u = np.r_[1:0:w * 1j]
            bar = cmap_ug(u, peak_f(u, params.mu_g, params.sigma_g))
            bar = bar[:, None].repeat(16, 1)
            vis_u = jp.hstack([bar, vis_u])
        vis = jp.hstack([vis_u, vis])
    return vis


fontpath = plt.matplotlib.get_data_path() + '/fonts/ttf/DejaVuSansMono.ttf'
pil_font = PIL.ImageFont.truetype(fontpath, size=16)


def text_overlay(img, text, pos=(20, 10), color=(255, 255, 255)):
    img = np2pil(img)
    draw = PIL.ImageDraw.Draw(img)
    draw.text(pos, text, fill=color, font=pil_font)
    return img


def animate_lenia(params, tracks, rate=10, slow_start=0, w=400, show_UG=True,
                  text=None, vid=None, bar_len=None,
                  bar_ofs=0, extent=None):
    if vid is None:
        vid = VideoWriter(fps=60)
    if extent is None:
        extent = jp.abs(tracks).max() * 1.2
    if bar_len is None:
        bar_len = len(tracks)
    for i, points in enumerate(tracks):
        if not (i < slow_start or i % rate == 0):
            continue
        img = show_lenia(params, points, extent, w=w, show_UG=show_UG)
        bar = np.linspace(0, bar_len, img.shape[1])
        bar = (0.5 + (bar >= i + bar_ofs)[:, None] * jp.ones(3) * 0.5)[None].repeat(2, 0)
        frame = jp.vstack([img, bar])
        if text is not None:
            frame = text_overlay(frame, text)
        res = vid(frame)

        if not res:
            break
    return vid


def odeint_euler(f, params, x0, dt, n: Optional[int]):
    def step_f(x, _):
        x = x + dt * f(params, x)
        return x, x

    return jax.lax.scan(step_f, x0, None, n)[1]


def point_fields_f(params, points):
    return jax.vmap(partial(fields_f, params, points))(points)


def total_energy_f(params, points):
    return point_fields_f(params, points).E.sum()


def total_motion_f(params, points):
    return -jax.grad(partial(total_energy_f, params))(points)


def calc_K_weight(mu, sigma, dim_n):
    r = jp.linspace(max(mu - sigma * 4, 0.0), mu + sigma * 4, 51)
    y = peak_f(r, mu, sigma) * r ** (dim_n - 1)
    s = jp.trapezoid(y, r) * {2: 2, 3: 4}[dim_n] * jp.pi
    return 1.0 / s


def create_params(m_k, s_k, m_g, s_g, rep, dim_n):
    w_k = calc_K_weight(m_k, s_k, dim_n)
    return Params(m_k, s_k, w_k, m_g, s_g, rep)


def test():
    params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    key = jax.random.PRNGKey(20)
    points0 = (jax.random.uniform(key, [200, 2]) - 0.5) * 12.0
    dt = 0.1

    rotor_story = odeint_euler(motion_f, params, points0, dt, 10000)
    animate_lenia(params, rotor_story)

    energy_log = jax.lax.map(partial(total_energy_f, params), rotor_story)
    plt.figure(figsize=(8, 4))
    plt.ylim(energy_log.min() - 10.0, 30.0)
    plt.plot(energy_log)
    plt.show()

if __name__ == '__main__':
    test()
