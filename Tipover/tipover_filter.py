import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_figure_defaults():
    plt.rcParams['lines.linewidth'] = 2    # linewidth
    plt.rcParams['lines.markersize'] = 2   # marker size

    plt.rcParams['axes.linewidth'] = 2     # linewidth
    plt.rcParams['axes.labelsize'] = 11    # axes font size
    plt.rcParams['xtick.labelsize'] = 11   # x-tick font size
    plt.rcParams['ytick.labelsize'] = 11   # y-tick font size

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 10          # size for text
    plt.rcParams['legend.fontsize'] = 11    # legend font size
    plt.rcParams['legend.title_fontsize'] = 11  # legend title font size

    plt.rcParams['axes.formatter.use_locale'] = False
    plt.rcParams['legend.handlelength'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # (if needed)

# terrain parameters
terrain_params = {'a1': 0.6, 'a2': 0.1, 'a3': 0.2,
    'f1': 0.2, 'f2': 0.3, 'f3': 0.5, 'f4': 0.7}

# 3D parametric terrain function
def terrain_function(x, y, params):
    """
    3D parametric terrain function
    """
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    f1, f2, f3, f4 = params['f1'], params['f2'], params['f3'], params['f4']

    return (a1 * np.sin(f1 * x) * np.cos(f1 * y)
            + a2 * x * np.cos(f2 * y)
            + a3 * np.sin(f3 * x) * np.cos(f4 * y))

# partial derivatives of the terrain
def terrain_slope(x, y, params):
    """
    Partial derivatives of the terrain
    """
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    f1, f2, f3, f4 = params['f1'], params['f2'], params['f3'], params['f4']

    dz_dx = (a1 * f1 * np.cos(f1 * x) * np.cos(f1 * y)
             + a2 * np.cos(f2 * y)
             + a3 * f3 * np.cos(f3 * x) * np.cos(f4 * y))

    dz_dy = (-a1 * f1 * np.sin(f1 * x) * np.sin(f1 * y)
             - a2 * x * f2 * np.sin(f2 * y)
             - a3 * np.sin(f3 * x) * f4 * np.sin(f4 * y))

    return dz_dx, dz_dy

# robot path based on the terrain
def terrain_path(timesteps, params):
    """
    Robot path based on the terrain
    """
    path_x = np.linspace(-20, 20, timesteps)
    path_y = 12 * np.sin(0.2 * path_x)
    path_z = terrain_function(path_x, path_y, params)
    return np.array([path_x, path_y, path_z])

# wrap to [-pi, pi]
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# 3D unicycle model on 3D terrain
def terrain_rhs(t, state, control_input, params):
    x, y, theta = state
    theta = wrap_to_pi(theta)
    v, omega = control_input

    # get beta and phi at current state
    dz_dx, dz_dy = terrain_slope(x, y, params)
    slope_heading = dz_dx * np.cos(theta) + dz_dy * np.sin(theta)
    slope_perp = -dz_dx * np.sin(theta) + dz_dy * np.cos(theta)
    beta = np.arctan(slope_heading)
    phi = np.arctan(slope_perp)
    beta = wrap_to_pi(beta)
    phi = wrap_to_pi(phi)

    cos_beta = np.cos(beta)
    cos_phi = np.cos(phi)

    x_dot = v * np.cos(theta) * cos_beta
    y_dot = v * np.sin(theta) * cos_beta
    theta_dot = omega * (cos_phi / cos_beta)
    return np.array([x_dot, y_dot, theta_dot])

# P-controller, u_nom
def P_control(state, goal, Kv, Kw):
    x, y, theta = state
    x_g, y_g = goal[:2]
    theta = wrap_to_pi(theta)

    p = np.array([x, y])
    p_g = np.array([x_g, y_g])

    ev = np.array([[np.cos(theta), np.sin(theta)]]).dot((p_g - p).reshape(-1, 1))
    e_perp_v = np.array([[-np.sin(theta), np.cos(theta)]]).dot((p_g - p).reshape(-1, 1))

    v = Kv * ev[0, 0]

    if np.allclose(p, p_g):
        omega = 0
    else:
        omega = Kw * np.arctan2(e_perp_v[0, 0], ev[0, 0])  

    v = np.clip(v, -20, 20)
    omega = np.clip(omega, -20, 20)

    return np.array([v, omega])

# Runge-Kutta (4th-order)
def rk4(rhs, t_span, s0):
    t0, tf = t_span
    dt = tf - t0
    s = s0
    k1 = rhs(t0, s)
    k2 = rhs(t0 + dt / 2, s + dt * k1 / 2)
    k3 = rhs(t0 + dt / 2, s + dt * k2 / 2)
    k4 = rhs(t0 + dt, s + dt * k3)
    s_next = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return [t0, tf], [s, s_next]

# ZOCBF safety function, state and input dependent CBF
def h(state, u, params, dt):
    b = 0.5
    h_cg = 1.0
    g = 9.81

    x_k, y_k, theta_k = state
    theta_k = wrap_to_pi(theta_k)

    dz_dx_k, dz_dy_k = terrain_slope(x_k, y_k, params)
    slope_perp_k = -dz_dx_k * np.sin(theta_k) + dz_dy_k * np.cos(theta_k)
    phi_k = np.arctan(slope_perp_k)

    phi_val = wrap_to_pi(phi_k)

    cos_phi = np.cos(phi_val)
    tan_phi = np.tan(phi_val)

    h_value = -np.abs(u[0] * u[1] / (g * cos_phi)) + b / (2 * h_cg) - tan_phi

    return h_value

# safety filter with ZOCBF
def filter_control_input(u_nom, u_prev, h_func, params, gamma, delta, rhs_func, state, dt):
    h_prev = h_func(state, u_prev, params, dt) # h(x, u^-)

    # if h(x,u^{-}) is negative, the system is already unsafe
    if h_prev < 0:
        logger.warning('System is unsafe at current state: h_prev: %f', h_prev)

    # constraint function, h(\phi(T;x,u), u) - h(x,u^{-}) \geq - \gamma(h(x,u^{-})) + \delta
    def zocbf_constraint(u):
        rhs_num = lambda t, s: rhs_func(t, s, u, params)
        _, state_rk4 = rk4(rhs_num, [0, dt], state)
        state_next = state_rk4[-1]  # next state from RK4
        h_next = h_func(state_next, u, params, dt)  # h(\phi(T;x,u), u)
        constraint_value = h_next - h_prev + gamma * h_prev - delta

        logger.debug('h_prev: %f, h_next: %f, constraint_value: %f', h_prev, h_next, constraint_value)

        return constraint_value 

    # cost function
    def objective(u):
        obj_value = np.linalg.norm(u - u_nom)**2
        
        return obj_value

    control_bounds = np.array([[-20, 20], [-20, 20]]) # input bound
    initial_guess = u_prev.copy() # initial guess is the previous control
    constraints = {'type': 'ineq', 'fun': zocbf_constraint}  # constraint dictionary

    # solve using 'SLSQP'
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=control_bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 500}
        )
    except Exception as e:
        logger.error('Optimization failed with exception: %s', e)
        result = None

    if result is not None and result.success:
        u_filtered = result.x
    else:
        u_filtered = u_nom  # nominal control if safety filter fails
        logger.warning('Filtering failed at state %s.', state)
        if result is not None:
            logger.warning('Optimization result: %s', result)

    return u_filtered

# sim with safety filter
def simulate_controller_rk4(controller, rhs_func, initial_state, Kv, Kw, params):
    state = initial_state.copy()
    states = [state]
    controls = []
    controls_filtered = []
    h_values = []

    path = terrain_path(timesteps, params)

    u_prev = np.zeros(2)  # u(0, x_0) = 0

    for i in range(timesteps):
        goal = [path[0][i], path[1][i], path[2][i]]  # Terrain path goal

        control_input = controller(state, goal, Kv, Kw) # P controller, u_nom
        controls.append(control_input)

        # safety filter, ZOCBF-OP
        control_input_filtered = filter_control_input(
            control_input, u_prev, h, params, gamma, delta, rhs_func, state, dt)

        u_prev = control_input_filtered # update u_prev

        h_val = h(state, u_prev, params, dt)
        h_values.append(float(h_val)) # for plotting

        # next state with safe control input
        rhs_num = lambda t, s: rhs_func(t, s, control_input_filtered, params)
        _, state_rk4 = rk4(rhs_num, [i * dt, (i + 1) * dt], state)
        state = state_rk4[-1]

        states.append(state)
        controls_filtered.append(control_input_filtered)

    return (np.array(states), np.array(controls), np.array(controls_filtered),
            np.array(h_values))

# sim without safety filter
def simulate_controller_rk4_nominal(controller, rhs_func, initial_state, Kv, Kw, params):
    state = initial_state.copy()
    states_nominal = [state]
    controls_nominal = []
    h_values_nominal = []

    path = terrain_path(timesteps, params)

    for i in range(timesteps):
        goal = [path[0][i], path[1][i], path[2][i]]

        control_input = controller(state, goal, Kv, Kw) # P controller, u_nom
        controls_nominal.append(control_input)

        h_val_nominal = h(state, control_input, params, dt)
        h_values_nominal.append(float(h_val_nominal))

        # next state with su_nom
        rhs_num = lambda t, s: rhs_func(t, s, control_input, params)
        _, state_rk4 = rk4(rhs_num, [i * dt, (i + 1) * dt], state)
        state = state_rk4[-1]

        states_nominal.append(state)
        # print(goal)
        # print(state)

    return (np.array(states_nominal), np.array(controls_nominal),
            np.array(h_values_nominal))

# parameters for sim
dt = 0.01
total_time = 20
timesteps = int(total_time / dt)

Kv = 1.5
Kw = 1.5

gamma = 0.05   # class-K
delta = 0.0001   # delta

path = terrain_path(timesteps, terrain_params) # path

theta_initial = np.arctan2(path[1][1] - path[1][0], path[0][1] - path[0][0])  # direction of the robot
initial_state = np.array([path[0][0], path[1][0], theta_initial]) # set initial state
 
# run sim for filtered control
(states_terrain, controls_terrain, controls_filtered, h_values) = simulate_controller_rk4(
    P_control, terrain_rhs, initial_state, Kv, Kw, terrain_params)

# run sim for u_nom control
(states_nominal, controls_nominal, h_values_nominal) = simulate_controller_rk4_nominal(
    P_control, terrain_rhs, initial_state, Kv, Kw, terrain_params)

# Plots
set_figure_defaults()

fig_width = 4.5
fig_height = fig_width / 1
fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)

# to make the edges of the box transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# plot terrain surface with higher resolution
x_res = np.linspace(-20, 20, int(total_time / dt))
y_res = np.linspace(-20, 20, int(total_time / dt))
X_res, Y_res = np.meshgrid(x_res, y_res)

Z_high_res = terrain_function(X_res, Y_res, terrain_params)  # pass params to terrain_function
ax.plot_surface(X_res, Y_res, Z_high_res, cmap='terrain', edgecolor='none', alpha=0.3)

ax.plot(path[0][:-100], path[1][:-100], path[2][:-100], linestyle='--', color='black', label='$x$, Ref.', linewidth=2)  # reference path

# trajectory with u_safe
closed_loop_filtered_z = terrain_function(states_terrain[:, 0], states_terrain[:, 1], terrain_params)
ax.plot(states_terrain[:, 0], states_terrain[:, 1], closed_loop_filtered_z, color='blue', label='$x$, ZOCBF', linewidth=2)

# split the path based on h_values_nominal
label_safe_added = False
label_unsafe_added = False
for i in range(len(states_nominal) - 1):
    x_segment = states_nominal[i:i+2, 0]
    y_segment = states_nominal[i:i+2, 1]
    z_segment = terrain_function(x_segment, y_segment, terrain_params)

    # h_values_nominal
    if h_values_nominal[i] < 0.005:
        path_color = 'red'
        path_style = '--'  
        if not label_unsafe_added:
            # ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2, label='$k_{nom}$, unsafe')
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2)

            label_unsafe_added = True
        else:
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2)

    else:
        path_color = 'green'
        path_style = '--'  
        if not label_safe_added:
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2, label='$x, k_{nom}$')
            label_safe_added = True
        else:
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2)

# add markers for start and end points
start_z = terrain_function(states_terrain[0, 0], states_terrain[0, 1], terrain_params)
end_z = terrain_function(states_terrain[-1, 0], states_terrain[-1, 1], terrain_params)
ax.scatter(states_terrain[0, 0], states_terrain[0, 1], start_z, color='black', marker='o', s=50, label='$x_{0}$')
ax.scatter(states_terrain[-1, 0], states_terrain[-1, 1], end_z, color='black', marker='*', s=50, label='$x_{f}$')

# tune view angle
ax.view_init(elev=50, azim=15)

ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-2, 3])

ax.set_box_aspect([1, 1, 1])  # equal scaling for all axes

ax.set_xlabel('$x^{\mathcal{I}}$')
ax.set_ylabel('$y^{\mathcal{I}}$')
ax.set_zlabel('$z^{\mathcal{I}}$')

ax.set_xticks([-20, 0, 20])
ax.set_yticks([-20, 0, 20])
ax.set_zticks([-2, 0, 3])

ax.tick_params(axis='x', which='major', labelsize=9)  # x-tick font size
ax.tick_params(axis='y', which='major', labelsize=9)  
ax.tick_params(axis='z', which='major', labelsize=9) 

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=6, fontsize=9, handletextpad=0.2, labelspacing=0.2, borderpad=0.2)

plt.tight_layout()
plt.savefig('terrainZOCBF.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot comparison of nominal and filtered control inputs
fig_width = 5.0
fig_height = fig_width / 1.3
fig = plt.figure(figsize=(fig_width, fig_height))

plt.subplot(2, 2, 1)
plt.plot(np.arange(timesteps) * dt, [c[0] for c in controls_filtered], label='$v$, ZOCBF', color='blue',)
plt.plot(np.arange(timesteps) * dt, [c[0] for c in controls_nominal], label='$v, k_{nom}$', color='green')
plt.xlabel('Time [s]')
plt.ylabel('$v$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=8)

plt.subplot(2, 2, 2)
plt.plot(np.arange(timesteps) * dt, [c[1] for c in controls_filtered], label='$\omega$, ZOCBF', color='blue',)
plt.plot(np.arange(timesteps) * dt, [c[1] for c in controls_nominal], label='$\omega, k_{nom}$', color='green')
plt.xlabel('Time [s]')
plt.ylabel('$\omega$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=8)

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(h_values)) * dt, h_values, label='$h$, ZOCBF', color='blue', linewidth=2)
plt.plot(np.arange(len(h_values_nominal)) * dt, h_values_nominal, label='$h, k_{nom}$', color='green', linewidth=2)
plt.xticks(np.arange(0, 21, 5))  
plt.xlabel('Time [s]')
plt.ylabel('$h(u, \phi)$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig('inputs_h.pdf', format='pdf', bbox_inches='tight')
plt.show()