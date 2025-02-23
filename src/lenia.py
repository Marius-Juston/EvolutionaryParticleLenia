import enum
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from util import cmap_e, cmap_ug, text_overlay

# Define types with frozen=True for immutability
Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep', defaults=(1.0,) * 6)
Fields = namedtuple('Fields', 'U G R E', defaults=(None,) * 4)


class IntegrationMethods(enum.Enum):
    EULER = 0  # Euler integration
    RK4 = 1  # Runge–Kutta
    RK45 = 2  # Runge–Kutta 4(5) adaptive timestep


Output = namedtuple("Output", "extent_scale w show_UG show_cmap fps", defaults=(1.5, 800, True, True, 60))

SimulationOptions = namedtuple("Simulation", "int_mode dt global_optimization ",
                               defaults=(IntegrationMethods.EULER, 0.1, False))


# Precompute RK45 constants (ideally defined once at module load)
RK45_B    = jnp.array([
    [0., 0., 0., 0., 0.],
    [1/4, 0., 0., 0., 0.],
    [3/32, 9/32, 0., 0., 0.],
    [1932/2197, -7200/2197, 7296/2197, 0., 0.],
    [439/216, -8., 3680/513, -845/4104, 0.],
    [-8/27, 2., -3544/2565, 1859/4104, -11/40]
])
RK45_C5   = jnp.array([16/135, 0., 6656/12825, 28561/56430, -9/50, 2/55])
RK45_C4   = jnp.array([25/216, 0., 1408/2565, 2197/4104, -1/5, 0.])

# Pad the Butcher tableau to fixed shape (6,6)
RK45_B_PAD = jnp.pad(RK45_B, ((0, 0), (0, 1)))  # shape (6,6)
# Precompute a lower triangular mask where element (i,j)=1 if j < i, else 0.
RK45_MASK = jnp.tril(jnp.ones((6,6), dtype=RK45_B.dtype), -1)

class ParticleLenia:
    def __init__(self, params: Params, sim_options: SimulationOptions, output: Output, points=None):
        int_funcs = {
            IntegrationMethods.EULER: self.step_f_euler,
            IntegrationMethods.RK4: self.step_f_rk4,
            IntegrationMethods.RK45: self.step_f_rk45
        }

        self.dt = sim_options.dt
        self.output_options = output
        self.sim_options = sim_options
        # Pre-compile the motion function based on optimization strategy
        self._f = jit(self.total_motion_f if self.sim_options.global_optimization else self.motion_f)
        self.step_f = jit(int_funcs[self.sim_options.int_mode])
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
    def step_f_euler(self, x: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, float]:
        """Single integration step"""
        next_x = x + dt * self._f(x)

        return next_x, dt

    @partial(jit, static_argnums=0)
    def step_f_rk4(self, x: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, float]:
        """
        RK4 integration step for more robust numerical integration.
        This allows for larger dt values while maintaining stability.

        The Runge-Kutta 4th order method uses four evaluations of the derivatives
        to compute a weighted average for better accuracy.
        """
        # Compute the four RK4 terms
        k1 = self._f(x)
        k2 = self._f(x + dt * k1 * 0.5)
        k3 = self._f(x + dt * k2 * 0.5)
        k4 = self._f(x + dt * k3)

        # Compute weighted average for the step
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update position using RK4 step
        next_x = x + dt * dx

        return next_x, dt

    @partial(jax.jit, static_argnums=0)
    def step_f_rk45(self, x: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, float]:
        """
        Highly optimized JAX-compatible adaptive Runge-Kutta-Fehlberg (RKF45) integration step.

        This implementation precomputes the RK45 coefficients and employs:
          - A padded Butcher tableau (RK45_B_PAD) with a static lower-triangular mask (RK45_MASK) to avoid dynamic slicing.
          - jax.lax.fori_loop for efficient, compiled iteration over the 6 stages.

        The routine computes both 5th order (x5) and 4th order (x4) estimates to determine an error
        for adaptive timestep adjustment.
        """
        # Tolerance and timestep bounds (could be parameters if needed)
        tol, min_dt, max_dt = 1e-6, 1e-6, 1.0

        # Evaluate the derivative at the initial state x
        k0 = self._f(x)
        k = jnp.zeros((6, *k0.shape), dtype=x.dtype)
        k = k.at[0].set(k0)

        def body(i, k):
            # Compute the weighted sum for the i-th stage:
            #   weighted_sum = dt * sum_{j=0}^{i-1} (b[i,j] * k[j])
            # Instead of dynamic slicing, use the fixed row RK45_B_PAD[i,:] and mask it with RK45_MASK[i,:].
            weighted_sum = dt * jnp.tensordot(RK45_B_PAD[i, :] * RK45_MASK[i, :], k, axes=([0], [0]))
            # Compute the intermediate state x_i using the weighted sum
            x_i = x + weighted_sum
            # Evaluate the derivative at x_i
            k_i = self._f(x_i)
            return k.at[i].set(k_i)

        # Loop over stages 1 to 5 using a compiled loop.
        k = jax.lax.fori_loop(1, 6, body, k)

        # Compute the 5th order solution (x5) and 4th order solution (x4)
        x5 = x + dt * jnp.tensordot(RK45_C5, k, axes=1)
        x4 = x + dt * jnp.tensordot(RK45_C4, k, axes=1)
        error = jnp.max(jnp.abs(x5 - x4))

        # Compute the new timestep based on the error estimate; the exponent 0.2 reflects the 5th order accuracy.
        dt_factor = jnp.power(tol / (error + 1e-10), 0.2)
        dt_new = jnp.clip(0.9 * dt * dt_factor, min_dt, max_dt)

        # Accept the new state if the error is within tolerance; otherwise, reject and reduce dt.
        accept = error <= tol
        next_x = jnp.where(accept, x5, x)
        next_dt = jnp.where(accept, dt_new, dt * 0.5)

        return next_x, next_dt

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

    @partial(jit, static_argnums=0)
    def show_lenia(self, points, extent):
        """Optimized visualization with batched computations"""
        # Use jnp.meshgrid for consistent GPU usage
        x, y = jnp.meshgrid(
            jnp.linspace(-extent, extent, self.output_options.w),
            jnp.linspace(-extent, extent, self.output_options.w)
        )
        xy = jnp.stack([x, y], axis=-1)

        # Vectorized field computations
        e0 = -self.peak_f(0.0, self.params.mu_g, self.params.sigma_g)
        fields = vmap(lambda pos: self.fields_f(points, pos))(xy.reshape(-1, 2))
        fields = jax.tree.map(lambda x: x.reshape(self.output_options.w, self.output_options.w), fields)

        # Optimized distance computation
        r2 = jnp.min(jnp.sum(jnp.square(xy[..., None, :] - points), axis=-1), axis=-1)
        points_mask = jnp.clip(r2 / 0.02, 0, 1.0)[..., None]

        # Visualization computations
        vis = cmap_e(fields.E - e0) * points_mask

        if self.output_options.show_cmap:
            e_mean = jnp.mean(vmap(lambda p: self.fields_f(points, p))(points).E)
            bar = jnp.linspace(0.5, -0.5, self.output_options.w)
            bar = cmap_e(bar) * (1.0 - self.peak_f(bar, e_mean - e0, 0.005)[:, None])
            vis = jnp.hstack([vis, jnp.repeat(bar[:, None], 16, axis=1)])

        if self.output_options.show_UG:
            vis_u = cmap_ug(fields.U, fields.G) * points_mask
            if self.output_options.show_cmap:
                u = jnp.linspace(1, 0, self.output_options.w)
                bar = cmap_ug(u, self.peak_f(u, self.params.mu_g, self.params.sigma_g))
                bar = jnp.repeat(bar[:, None], 16, axis=1)
                vis_u = jnp.hstack([bar, vis_u])
            vis = jnp.hstack([vis_u, vis])

        return vis

    def animate_lenia(self, vid=None):
        """Animation with optimized computation"""
        if vid is None:
            from video import VideoWriter
            vid = VideoWriter(fps=self.output_options.fps)

        with vid:
            i = 0
            while True:
                extent = jnp.max(jnp.abs(self.points)) * self.output_options.extent_scale

                self.points, self.dt = self.step_f(self.points, self.dt)
                img = self.show_lenia(self.points, extent)

                if self.sim_options.int_mode == IntegrationMethods.RK45:
                    img = text_overlay(img, f"dt: {self.dt:.2f}")

                if not vid(img):
                    break

                i += 1


if __name__ == '__main__':
    params = Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)
    output_options = Output()
    sim_options = SimulationOptions(int_mode=IntegrationMethods.RK45, dt=10)

    l = ParticleLenia(params, sim_options, output_options)

    l.animate_lenia()
