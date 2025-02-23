from collections import namedtuple
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jp
import numpy as np
from jax import jit

from util import cmap_e, vmap2, cmap_ug
from video import VideoWriter

Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')


class ParticleLenia:
    def __init__(self, params: Params, global_optimization=False, dt=0.1, points=None):
        self.dt = dt
        self._f: Callable[jax.Array, jax.Array] = self.total_motion_f if global_optimization else self.motion_f

        self.params = params

        if points:
            self.points = points
        else:
            key = jax.random.PRNGKey(20)
            self.points = (jax.random.uniform(key, [200, 2]) - 0.5) * 12.0

    @jit
    def peak_f(self, x: jax.Array, mu: float, sigma: float) -> jax.Array:
        """
        The growth function (Gaussian Distribution)
        :param x:
        :param mu:
        :param sigma:
        :return:
        """
        return jp.exp(-((x - mu) / sigma) ** 2)

    @jit
    def fields_f(self, points: jax.Array, x: jax.Array) -> Fields:
        """
        The big function representing the potential field of the system
        :param points:
        :param x:
        :return:
        """
        r = jp.sqrt(jp.square(x - points).sum(-1).clip(1e-10))
        U = self.peak_f(r, self.params.mu_k, self.params.sigma_k).sum() * self.params.w_k
        G = self.peak_f(U, self.params.mu_g, self.params.sigma_g)
        R = self.params.c_rep / 2 * ((1.0 - r).clip(0.0) ** 2).sum()
        return Fields(U, G, R, E=R - G)

    @jit
    def motion_f(self, points: jax.Array) -> jax.Array:
        """
        Given the energy field take the gradient and then to minimize move to the inverse gradient
        :return:
        """
        grad_E = jax.grad(lambda x: self.fields_f(points, x).E)
        return -jax.vmap(grad_E)(points)

    @jit
    def odeint_euler(self, x0: jax.Array, n: int) -> jax.Array:
        """
        Set Euler integration of the particles
        :param x0:
        :param n:
        :return:
        """
        return jax.lax.scan(self.step_f, x0, None, n)[1]

    @jit
    def step_f(self, x: jax.Array, _):
        x = x + self.dt * self._f(x)
        return x, x

    @jit
    def point_fields_f(self, points: jax.Array) -> jax.Array:
        """
        Computes the particle wise energy of the system
        :return:
        """
        return jax.vmap(partial(self.fields_f, points))(points)

    @jit
    def total_energy_f(self, points: jax.Array) -> jax.Array:
        """
        The sum total energy in the system
        :return:
        """
        return self.point_fields_f(points).E.sum()

    @jit
    def total_motion_f(self, points: jax.Array) -> jax.Array:
        """
        The gradient of the total energy of the system per point (used to minimize in a global sense rather than a local greedy way)
        :param points:
        :return:
        """
        return -jax.grad(self.total_energy_f)(points)

    @partial(jax.jit, static_argnames=['w', 'show_UG', 'show_cmap'])
    def show_lenia(self, points, extent, w=400, show_UG=False, show_cmap=True):
        xy = jp.mgrid[-1:1:w * 1j, -1:1:w * 1j].T * extent
        e0 = -self.peak_f(0.0, self.params.mu_g, self.params.sigma_g)
        f = partial(self.fields_f, points)
        fields = vmap2(f)(xy)
        r2 = jp.square(xy[..., None, :] - points).sum(-1).min(-1)
        points_mask = (r2 / 0.02).clip(0, 1.0)[..., None]
        vis = cmap_e(fields.E - e0) * points_mask
        if show_cmap:
            e_mean = jax.vmap(f)(points).E.mean()
            bar = np.r_[0.5:-0.5:w * 1j]
            bar = cmap_e(bar) * (1.0 - self.peak_f(bar, e_mean - e0, 0.005)[:, None])
            vis = jp.hstack([vis, bar[:, None].repeat(16, 1)])
        if show_UG:
            vis_u = cmap_ug(fields.U, fields.G) * points_mask
            if show_cmap:
                u = np.r_[1:0:w * 1j]
                bar = cmap_ug(u, self.peak_f(u, self.params.mu_g, self.params.sigma_g))
                bar = bar[:, None].repeat(16, 1)
                vis_u = jp.hstack([bar, vis_u])
            vis = jp.hstack([vis_u, vis])
        return vis

    def run_lenia(self):

        while True:
            self.points = self.step_f(self.points, None)[0]
            yield self.points

    def animate_lenia(self, rate=10, slow_start=0, w=400, show_UG=True,
                      vid=None, extent=None) -> None:
        if vid is None:
            vid = VideoWriter(fps=60)

        i = 0

        points = self.step_f(self.points, None)[0]

        with vid:
            # for points in  self.run_lenia():
            while True:
                if not (i < slow_start or i % rate == 0):
                    continue

                if extent is None:
                    extent = jp.abs(points).max() * 1.2

                points = self.step_f(points, None)[0]

                img = self.show_lenia(points, extent, w=w, show_UG=show_UG)

                res = vid(img)

                if not res:
                    break

                i += 1


if __name__ == '__main__':
    params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    l = ParticleLenia(params, global_optimization=True)

    l.animate_lenia()
