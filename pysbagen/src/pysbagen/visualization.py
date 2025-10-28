import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.special import jn, jnp_zeros

def bessel_derivative_zero(n, m):
    if n == 0:
        zeros = list(jnp_zeros(n, m + 1))
        zeros.insert(0, 0)
        return zeros[m]
    else:
        return jnp_zeros(n, m + 1)[-1]

def f(r, theta):
    return r * np.cos(theta)

def a_nm(n, m, a):
    z_nm = bessel_derivative_zero(n, m)
    numerator = integrate.dblquad(lambda theta, r: jn(n, z_nm * r * 1.0/a) * np.cos(n * theta) * f(r, theta) * r, 0, a, lambda r: 0, lambda r: 2 * np.pi)[0]
    denominator = integrate.quad(lambda r: jn(n, z_nm * r * 1.0/a) ** 2 * r, 0, a)[0]
    return numerator / (2 * np.pi * denominator)

def b_nm(n, m, a):
    z_nm = bessel_derivative_zero(n, m)
    numerator = integrate.dblquad(lambda theta, r: jn(n, z_nm * r * 1.0/a) * np.sin(n * theta) * f(r, theta) * r, 0, a, lambda r: 0, lambda r: 2 * np.pi)[0]
    denominator = integrate.quad(lambda r: jn(n, z_nm * r * 1.0/a) ** 2 * r, 0, a)[0]
    return numerator / (2 * np.pi * denominator)

def u(n, m, r, theta, a):
    z_nm = bessel_derivative_zero(n, m)
    A_nm = a_nm(n, m, a)
    B_nm = b_nm(n, m, a)
    return jn(n, z_nm * r * 1.0/a) * (A_nm * np.cos(n * theta) + B_nm * np.sin(n * theta))

def generate_chladni_pattern(n, m, a=1.0, resolution=100):
    r = np.linspace(0, a, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    R, Theta = np.meshgrid(r, theta)

    Z = u(n, m, R, Theta, a)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.contourf(Theta, R, Z, levels=100, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    return fig

def generate_pattern_grid(n_max, m_max):
    fig, axs = plt.subplots(m_max, n_max, figsize=(1.4 * n_max, 1.2 * m_max), subplot_kw={'projection': 'polar'})

    for m in range(m_max):
        for n in range(n_max):
            Z = u(n, m, *np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 2 * np.pi, 100)), a=1.0)
            ax = axs[m, n] if m_max > 1 else axs[n]
            ax.contour(np.linspace(0, 2 * np.pi, 100), np.linspace(0, 1, 100), Z, levels=[0], colors='black', linewidths=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"({n}, {m})")

    plt.tight_layout()
    return fig

# A list of (n, m) pairs that are known to produce interesting patterns.
# This list can be expanded.
GOOD_PAIRS = [
    (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3),
    (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (6, 1), (6, 2), (7, 1),
]

def map_freq_to_params(freq):
    """Maps a frequency to a pair of (n, m) parameters."""
    # This is a simple mapping for now. It can be improved.
    index = int(freq / 20) % len(GOOD_PAIRS)
    return GOOD_PAIRS[index]

if __name__ == '__main__':
    # Example usage:
    fig = generate_pattern_grid(8, 8)
    plt.show()
