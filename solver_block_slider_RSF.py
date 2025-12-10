import numpy as np
import matplotlib.pyplot as plt

# ==========================
# MODEL PARAMETERS
# ==========================

# Normal stress and RSF parameters
sigmaN = 10e6       # [Pa]
a      = 0.08
b      = 0.15
D_c    = 1e-6       # Dc [m]

# Reference values for the RSF law (mu_* and V_*)
mu_star = 0.6       # reference friction μ_*
V_star  = 1e-6      # reference slip velocity V_* [m/s]

# Loading velocity
Vpl = 1e-7          # [m/s]

# === Critical stiffness and spring stiffness ===
kc = sigmaN * (b - a) / D_c
k  = 0.8 * kc       # k < kc to enable stick-slip

# Effective mass per unit area
M = 1e8             # [kg/m^2]

# Radiation damping
eta = 1e10          # [Pa·s/m]

# Total simulation time
t0 = 0.0
tf = 200.0          # [s]

# Newton and fixed-point parameters
max_newton_iter = 40
tol_newton      = 1e-10

max_fp_iter = 25
tol_fp      = 1e-4   # fixed-point tolerance for θ (log scale)

# Numerical thresholds
V_min = 1e-14        # avoids issues in θ when v → 0

# Adaptive time step parameters
dt_init = 1e-2       # initial time step
dt_min  = 1e-5
dt_max  = 5e-1

delta_tol = 5e-3     # step-doubling tolerance (log θ)


# ==========================
# RSF FRICTION LAW
# ==========================
def mu_rsf_rbz(V, theta):
    """
    Regularized RSF law of Rice–Ben-Zion:

        μ(V, θ) = a * asinh( V / (2 V_theta) )

    where
        V_theta = V_star * exp( -(mu_star + b * log(theta * V_star / Dc)) / a )

    V: slip velocity magnitude |v|.
    """
    theta = max(theta, 1e-20)

    V_theta = V_star * np.exp(-(mu_star + b * np.log(theta * V_star / D_c)) / a)
    mu      = a * np.arcsinh(V / (2.0 * V_theta))
    return mu


# ==========================
# ANALYTICAL UPDATE OF θ
# ==========================
def update_theta_closed(theta_prev, v_prev_abs, v_n_abs, dt):
    """
    Integrates the state evolution law
        dθ/dt = 1 - |v| θ / D_c
    assuming |v| is constant over the step and using
        |v|_{n-1/2} = (|v_{n-1}| + |v_n|) / 2.
    """
    v_half = 0.5 * (v_prev_abs + v_n_abs)

    if v_half < V_min:
        # nearly pure stick → dθ/dt ≈ 1
        return theta_prev + dt

    theta_ss = D_c / v_half
    exp_fac  = np.exp(-v_half * dt / D_c)
    theta_n  = theta_ss + (theta_prev - theta_ss) * exp_fac
    return theta_n


# ==========================
# NEWTON SOLVER FOR v_n (1D version of Eq. (15))
# ==========================
def newton_velocity(u_prev, v_prev, theta_n, t_n, dt):
    """
    Solves for v_n the implicit equation (adapted from Eq. (15)):

        A_n v_n + τ_fric(v_n, θ_n) = R_n

    where:
        A_n = 2M/dt + eta + (dt/2) k
        R_n = (2M/dt + eta - (dt/2) k) v_{n-1}
              - k u_{n-1} + k Vpl t_n

        τ_fric(v,θ) = sigmaN * μ(|v|,θ) * sign(v)
    """

    # Coefficients A_n and R_n
    A = (2.0 * M / dt) + eta + 0.5 * dt * k
    R = ((2.0 * M / dt) + eta - 0.5 * dt * k) * v_prev \
        - k * u_prev + k * Vpl * t_n

    # Initial guess: previous velocity
    v = v_prev

    for _ in range(max_newton_iter):

        V = abs(v)
        mu = mu_rsf_rbz(V, theta_n)
        tau_fric = sigmaN * mu * (np.sign(v) if V > 0 else 0.0)

        # Residual
        F = A * v + tau_fric - R

        # Numerical derivative dF/dv
        dv = 1e-6 * max(1.0, abs(v))
        v_pert = v + dv
        Vp = abs(v_pert)
        mu_p = mu_rsf_rbz(Vp, theta_n)
        tau_fric_p = sigmaN * mu_p * (np.sign(v_pert) if Vp > 0 else 0.0)

        Fp = A * v_pert + tau_fric_p - R
        dFdv = (Fp - F) / dv

        # Newton update
        v_new = v - F / dFdv

        if abs(v_new - v) < tol_newton * max(1.0, abs(v)):
            return v_new

        v = v_new

    # If Newton does not converge, return last iterate
    return v


# ==========================
# ONE TIME STEP (fixed dt) WITH FIXED-POINT ITERATION IN (θ, v)
# ==========================
def one_step_fixed_point(t, u_prev, v_prev, theta_prev, dt):
    """
    Performs ONE time step of size dt, from t to t+dt,
    solving the coupled system using a fixed-point iteration:

        θ^{k-1}_n  --(Newton)-->  v^{k}_n
        v^{k}_n    --(analytic)-> θ^{k}_n

    Returns: u_n, v_n, theta_n
    """
    t_n = t + dt

    # Initial prediction for θ_n (Eq. (18): θ_n^0 = θ_{n-1})
    theta_k = theta_prev
    v_n     = v_prev

    for _ in range(max_fp_iter):

        # Given θ_n^{k-1}, solve for v_n^{k} with Newton
        v_n = newton_velocity(u_prev, v_prev, theta_k, t_n, dt)

        # Update θ_n^{k} using the closed-form solution
        theta_new = update_theta_closed(theta_prev,
                                        abs(v_prev),
                                        abs(v_n),
                                        dt)

        # Convergence criterion: d(θ^{k}, θ^{k-1}) <= ε  (Eq. (19))
        if theta_new > 0.0 and theta_k > 0.0:
            d = abs(np.log(theta_new / theta_k))
        else:
            d = abs(theta_new - theta_k)

        theta_k = theta_new

        if d <= tol_fp:
            break

    theta_n = theta_k

    # Update displacement using trapezoidal/Newmark:
    #    u_n = u_{n-1} + dt/2 * (v_{n-1} + v_n)
    u_n = u_prev + 0.5 * dt * (v_prev + v_n)

    return u_n, v_n, theta_n


# ==========================
# INITIALIZATION
# ==========================
t = t0

# approximate steady-state: v ≈ Vpl,  θ_ss = D_c / Vpl
u0     = 0.0
v0     = Vpl
theta0 = D_c / Vpl

u = u0
v = v0
theta = theta0

# store history
t_vals     = [t]
u_vals     = [u]
v_vals     = [v]
theta_vals = [theta]
dt_vals    = []

dt = dt_init



# ==========================
# TIME LOOP WITH STEP-DOUBLING
# ==========================
while t < tf:

    # adjust dt if we overshoot the final time
    if t + dt > tf:
        dt = tf - t

    # STEP-DOUBLING: compare 1 large step vs 2 small steps
    while True:

        # (A) one step of size dt
        u1, v1, theta1 = one_step_fixed_point(t, u, v, theta, dt)

        # (B) two steps of size dt/2
        dt_half = 0.5 * dt
        u_half, v_half, theta_half = one_step_fixed_point(t, u, v, theta, dt_half)
        u2,    v2,    theta2      = one_step_fixed_point(t + dt_half,
                                                          u_half, v_half, theta_half,
                                                          dt_half)

        # distance between the two estimates of θ  (Eq. (14) in 1D)
        if theta1 > 0.0 and theta2 > 0.0:
            d = abs(np.log(theta2 / theta1))
        else:
            d = abs(theta2 - theta1)

        # refine the step if the error is too large
        if (d > delta_tol) and (dt > dt_min):
            dt *= 0.5
            continue
        else:
            # accept the finer (two-step) solution
            t_new     = t + dt
            u_new     = u2
            v_new     = v2
            theta_new = theta2
            break

    # optional coarsening: enlarge dt if error is very small
    if d < 0.25 * delta_tol and (2.0 * dt <= dt_max):
        dt *= 2.0

    # store accepted state
    t      = t_new
    u      = u_new
    v      = v_new
    theta  = theta_new

    t_vals.append(t)
    u_vals.append(u)
    v_vals.append(v)
    theta_vals.append(theta)
    dt_vals.append(dt)


# ==========================
# POSTPROCESSING: STRESSES AND PLOTS
# ==========================
t_vals     = np.array(t_vals)
u_vals     = np.array(u_vals)
v_vals     = np.array(v_vals)
theta_vals = np.array(theta_vals)

# friction coefficient and stresses
mu_vals  = np.array([mu_rsf_rbz(abs(vi), thi) for vi, thi in zip(v_vals, theta_vals)])
tau_fric = sigmaN * mu_vals               # [Pa]
tau_load = k * (Vpl * t_vals - u_vals)    # [Pa]


# |v(t)| in log scale
plt.figure(figsize=(10,4))
plt.semilogy(t_vals, np.abs(v_vals))
plt.xlabel("Time [s]")
plt.ylabel("|v(t)| [m/s]")
plt.title("Quasi-dynamic RSF Spring–Slider (stick-slip, adaptive dt)")
plt.grid(True)
plt.tight_layout()
plt.savefig("velocity_vs_time.png", dpi=300)

# τ_fric(t) and τ_load(t)
plt.figure(figsize=(10,4))
plt.plot(t_vals, tau_fric/1e6, label="τ_fric")
# plt.plot(t_vals, tau_load/1e6, "--", label="τ_load")
plt.xlabel("Time [s]")
plt.ylabel("τ(t) [MPa]")
plt.title("Time evolution of shear stress (friction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tau_vs_time_total.png", dpi=300)

# Zoom into a selected time window (adjust limits as needed)
t1_zoom, t2_zoom = 110.0, 122.0
mask = (t_vals >= t1_zoom) & (t_vals <= t2_zoom)

plt.figure(figsize=(6,4))
plt.plot(t_vals[mask], tau_fric[mask]/1e6, "-o", ms=2)
plt.xlabel("Time [s]")
plt.ylabel("τ_fric(t) [MPa]")
plt.title("Zoom: single stress-drop event")
plt.grid(True)
plt.tight_layout()
plt.savefig("tau_vs_time_zoom.png", dpi=300)