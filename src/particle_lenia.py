from collections import namedtuple
from functools import partial
from typing import Optional

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit

from util import vmap2, cmap_e, cmap_ug, text_overlay
from video import VideoWriter

plt.rcParams.update({"axes.grid": True})

Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')


@jit
def peak_f(x, mu, sigma):
    return jp.exp(-((x - mu) / sigma) ** 2)

@jit
def fields_f(p: Params, points, x):
    r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
    U = peak_f(r, p.mu_k, p.sigma_k).sum() * p.w_k
    G = peak_f(U, p.mu_g, p.sigma_g)
    R = p.c_rep / 2 * ((1.0 - r).clip(0.0) ** 2).sum()
    return Fields(U, G, R, E=R - G)

@jit
def motion_f(params, points):
    grad_E = jax.grad(lambda x: fields_f(params, points, x).E)
    return -jax.vmap(grad_E)(points)


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

@jit
def point_fields_f(params, points):
    return jax.vmap(partial(fields_f, params, points))(points)

@jit
def total_energy_f(params, points):
    return point_fields_f(params, points).E.sum()

@jit
def total_motion_f(params, points):
    return -jax.grad(partial(total_energy_f, params))(points)


def test():
    params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    key = jax.random.PRNGKey(20)
    points0 = (jax.random.uniform(key, [200, 2]) - 0.5) * 12.0
    dt = 0.1

    rotor_story = odeint_euler(motion_f, params, points0, dt, 10000)
    animate_lenia(params, rotor_story, w=800)

    energy_log = jax.lax.map(partial(total_energy_f, params), rotor_story)
    plt.figure(figsize=(8, 4))
    plt.ylim(energy_log.min() - 10.0, 30.0)
    plt.plot(energy_log)
    plt.show()


if __name__ == '__main__':
    test()
