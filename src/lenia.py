from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

# Define types with frozen=True for immutability
Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep', defaults=(None,) * 6)
Fields = namedtuple('Fields', 'U G R E', defaults=(None,) * 4)


class ParticleLenia:
    def __init__(self, params: Params, global_optimization=False, dt=0.1, points=None):
        self.dt = dt
        # Pre-compile the motion function based on optimization strategy
        self._f = jit(self.total_motion_f if global_optimization else self.motion_f)
        self.params = params

        if points is not None:
            self.points = jnp.array(points)
        else:
            key = jax.random.PRNGKey(20)
            # Use jnp.array for consistent GPU usage
            self.points = (jax.random.uniform(key, (200, 2)) - 0.5) * 12.0

    @staticmethod
    @jit
    def peak_f(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
        """Vectorized Gaussian distribution computation"""
        return jnp.exp(-jnp.square((x - mu) / sigma))

    @partial(jit, static_argnums=0)
    def fields_f(self, points: jnp.ndarray, x: jnp.ndarray) -> Fields:
        """Compute field potentials using vectorized operations"""
        # Vectorized distance calculation
        diff = x - points
        r = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1).clip(1e-10))

        # Vectorized field computations
        U = jnp.sum(self.peak_f(r, self.params.mu_k, self.params.sigma_k)) * self.params.w_k
        G = self.peak_f(U, self.params.mu_g, self.params.sigma_g)
        R = self.params.c_rep * 0.5 * jnp.sum(jnp.square(jnp.clip(1.0 - r, 0.0, None)))

        return Fields(U=U, G=G, R=R, E=R - G)

    @partial(jit, static_argnums=0)
    def motion_f(self, points: jnp.ndarray) -> jnp.ndarray:
        """Compute motion vectors using vectorized gradient"""
        grad_E = grad(lambda x: self.fields_f(points, x).E)
        return -vmap(grad_E)(points)

    @partial(jit, static_argnums=(0, 2))
    def odeint_euler(self, x0: jnp.ndarray, n: int) -> jnp.ndarray:
        """Vectorized Euler integration"""

        def scan_fn(x, _):
            next_x = x + self.dt * self._f(x)
            return next_x, next_x

        return jax.lax.scan(scan_fn, x0, None, n)[1]

    @partial(jit, static_argnums=0)
    def step_f(self, x: jnp.ndarray, _):
        """Single integration step"""
        next_x = x + self.dt * self._f(x)
        return next_x, next_x

    @partial(jit, static_argnums=0)
    def point_fields_f(self, points: jnp.ndarray) -> jnp.ndarray:
        """Vectorized computation of per-point fields"""
        return vmap(partial(self.fields_f, points))(points)

    @partial(jit, static_argnums=0)
    def total_energy_f(self, points: jnp.ndarray) -> jnp.ndarray:
        """Compute total system energy"""
        return jnp.sum(self.point_fields_f(points).E)

    @partial(jit, static_argnums=0)
    def total_motion_f(self, points: jnp.ndarray) -> jnp.ndarray:
        """Global optimization using total energy gradient"""
        return -grad(self.total_energy_f)(points)

    @partial(jit, static_argnums=(0, 3, 4, 5))
    def show_lenia(self, points, extent, w=400, show_UG=False, show_cmap=True):
        """Optimized visualization with batched computations"""
        # Use jnp.meshgrid for consistent GPU usage
        x, y = jnp.meshgrid(
            jnp.linspace(-extent, extent, w),
            jnp.linspace(-extent, extent, w)
        )
        xy = jnp.stack([x, y], axis=-1)

        # Vectorized field computations
        e0 = -self.peak_f(0.0, self.params.mu_g, self.params.sigma_g)
        fields = vmap(lambda pos: self.fields_f(points, pos))(xy.reshape(-1, 2))
        fields = jax.tree.map(lambda x: x.reshape(w, w), fields)

        # Optimized distance computation
        r2 = jnp.min(jnp.sum(jnp.square(xy[..., None, :] - points), axis=-1), axis=-1)
        points_mask = jnp.clip(r2 / 0.02, 0, 1.0)[..., None]

        # Visualization computations
        from util import cmap_e, cmap_ug  # Import visualization utilities
        vis = cmap_e(fields.E - e0) * points_mask

        if show_cmap:
            e_mean = jnp.mean(vmap(lambda p: self.fields_f(points, p))(points).E)
            bar = jnp.linspace(0.5, -0.5, w)
            bar = cmap_e(bar) * (1.0 - self.peak_f(bar, e_mean - e0, 0.005)[:, None])
            vis = jnp.hstack([vis, jnp.repeat(bar[:, None], 16, axis=1)])

        if show_UG:
            vis_u = cmap_ug(fields.U, fields.G) * points_mask
            if show_cmap:
                u = jnp.linspace(1, 0, w)
                bar = cmap_ug(u, self.peak_f(u, self.params.mu_g, self.params.sigma_g))
                bar = jnp.repeat(bar[:, None], 16, axis=1)
                vis_u = jnp.hstack([bar, vis_u])
            vis = jnp.hstack([vis_u, vis])

        return vis

    def run_lenia(self):
        """Generator for simulation steps"""
        points = self.points
        while True:
            points = self.step_f(points, None)[0]
            yield points

    def animate_lenia(self, w=400, show_UG=True,
                      vid=None, fps=60):
        """Animation with optimized computation"""
        if vid is None:
            from video import VideoWriter
            vid = VideoWriter(fps=fps)

        self.points = self.step_f(self.points, None)[0]

        with vid:
            i = 0
            while True:
                extent = jnp.max(jnp.abs(self.points)) * 1.5

                self.points = self.step_f(self.points, None)[0]
                img = self.show_lenia(self.points, extent, w=w, show_UG=show_UG)

                if not vid(img):
                    break

                i += 1


if __name__ == '__main__':
    params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    l = ParticleLenia(params, global_optimization=False)

    l.animate_lenia()
