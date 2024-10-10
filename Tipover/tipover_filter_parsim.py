import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
import jax
import jax.numpy as jnp
import timeit
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_figure_defaults():
    plt.rcParams['lines.linewidth'] = 2    # linewidth
    plt.rcParams['lines.markersize'] = 2   # marker size

    plt.rcParams['axes.linewidth'] = 1     # linewidth
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

def terrain_function_jax(x, y, params):
    """
    3D parametric terrain function, now using JAX
    """
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    f1, f2, f3, f4 = params['f1'], params['f2'], params['f3'], params['f4']

    return (a1 * jnp.sin(f1 * x) * jnp.cos(f1 * y)
            + a2 * x * jnp.cos(f2 * y)
            + a3 * jnp.sin(f3 * x) * jnp.cos(f4 * y))

# partial derivatives of the terrain
def terrain_slope_jax(x, y, params):
    """
    Partial derivatives of the terrain, now using JAX auto-diff
    """
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    f1, f2, f3, f4 = params['f1'], params['f2'], params['f3'], params['f4']

    dz_dx = jax.grad(terrain_function_jax, argnums=0)(x, y, params)

    dz_dy = jax.grad(terrain_function_jax, argnums=1)(x, y, params)

    # dz_dx = (a1 * f1 * jnp.cos(f1 * x) * jnp.cos(f1 * y)
    #          + a2 * jnp.cos(f2 * y)
    #          + a3 * f3 * jnp.cos(f3 * x) * jnp.cos(f4 * y))

    # dz_dy = (-a1 * f1 * jnp.sin(f1 * x) * jnp.sin(f1 * y)
    #          - a2 * x * f2 * jnp.sin(f2 * y)
    #          - a3 * jnp.sin(f3 * x) * f4 * jnp.sin(f4 * y))

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

def wrap_to_pi_jax(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

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

def terrain_rhs_jax(t, state, control_input, params):
    x, y, theta = state
    theta = wrap_to_pi_jax(theta)
    v, omega = control_input

    # Get beta and phi at the current state
    dz_dx, dz_dy = terrain_slope_jax(x, y, params)
    slope_heading = dz_dx * jnp.cos(theta) + dz_dy * jnp.sin(theta)
    slope_perp = -dz_dx * jnp.sin(theta) + dz_dy * jnp.cos(theta)
    beta = jnp.arctan(slope_heading)
    phi = jnp.arctan(slope_perp)
    beta = wrap_to_pi_jax(beta)
    phi = wrap_to_pi_jax(phi)

    cos_beta = jnp.cos(beta)
    cos_phi = jnp.cos(phi)

    x_dot = v * jnp.cos(theta) * cos_beta
    y_dot = v * jnp.sin(theta) * cos_beta
    theta_dot = omega * (cos_phi / cos_beta)
    return jnp.array([x_dot, y_dot, theta_dot])

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

def h_jax(state, u, params, dt):
    b = 0.5
    h_cg = 1.0
    g = 9.81

    x_k, y_k, theta_k = state
    theta_k = wrap_to_pi_jax(theta_k)

    dz_dx_k, dz_dy_k = terrain_slope_jax(x_k, y_k, params)
    slope_perp_k = -dz_dx_k * jnp.sin(theta_k) + dz_dy_k * jnp.cos(theta_k)
    phi_k = jnp.arctan(slope_perp_k)

    phi_val = wrap_to_pi_jax(phi_k)

    cos_phi = jnp.cos(phi_val)
    tan_phi = jnp.tan(phi_val)

    h_value = -jnp.abs(u[0] * u[1] / (g * cos_phi)) + b / (2 * h_cg) - tan_phi

    return h_value

# Generate a uniform grid
def create_uniform_grid(bounds_min, bounds_max, grid_points_per_dim):
    # Generate equally spaced points in each dimension
    grid_axes = [np.linspace(bounds_min[i], bounds_max[i], grid_points_per_dim[i]) 
                 for i in range(bounds_min.shape[0])]
    # Create a meshgrid from the points
    mesh = np.meshgrid(*grid_axes, indexing='ij')
    # Reshape meshgrid to list of points
    grid = np.stack(mesh, axis=-1).reshape(-1, bounds_min.shape[0])
    return grid


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


# safety filter with ZOCBF, parallel sim without JAX
def filter_control_input_parsim(u_nom, u_prev, h_func, params, gamma, delta, rhs_func, state, dt,grid_u):
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

    
    try:
        ## using map to avoid loops for better efficiency, without using jax 
        # zocbf_constraint_values = np.array(list(map(zocbf_constraint,grid_u)))
        # safe_inputs = list(filter(lambda uv: uv[1]>=0, zip(grid_u,zocbf_constraint_values)))
        # safe_inputs = [u for u,_ in safe_inputs]

        # cost_values = np.array(list(map(objective,safe_inputs)))
        # min_cost_ind = np.argmin(cost_values)
        # u_filtered  = safe_inputs[min_cost_ind]

        ## using one loop for better efficiency, without using jax 
        # Initialize variables for tracking the best solution
        min_cost = np.inf
        best_u = None

        # Iterate over grid_u, filtering and computing the cost in one pass
        for u in grid_u:
            zocbf_value = zocbf_constraint(u)
            if zocbf_value >= 0:  # Only evaluate if the constraint is satisfied
                cost = objective(u)
                if cost < min_cost:  # Update the best cost and best u
                    min_cost = cost
                    best_u = u
                if min_cost<1e-6:
                    break

        # The best input is stored in `best_u`
        u_filtered = best_u

    except Exception as e:
        logger.error('Optimization failed with exception: %s', e)
        u_filtered = u_nom  # nominal control if safety filter fails
        logger.warning('Filtering failed at state %s.', state)

    return u_filtered  

# safety filter with ZOCBF, parallel simulation with JAX
def filter_control_input_parsim_jax(u_nom, u_prev, h_func, params, gamma, delta, rhs_func, state, dt,grid_u):
    h_prev = h_func(state, u_prev, params, dt) # h(x, u^-)

    # if h(x,u^{-}) is negative, the system is already unsafe
    # if h_prev < 0:
    #     logger.warning('System is unsafe at current state: h_prev: %f', h_prev)

    # constraint function, h(\phi(T;x,u), u) - h(x,u^{-}) \geq - \gamma(h(x,u^{-})) + \delta
    def zocbf_constraint(u):
        rhs_num = lambda t, s: rhs_func(t, s, u, params)
        _, state_rk4 = rk4(rhs_num, [0, dt], state)
        state_next = state_rk4[-1]  # next state from RK4
        h_next = h_func(state_next, u, params, dt)  # h(\phi(T;x,u), u)
        constraint_value = h_next - h_prev + gamma * h_prev - delta

        # logger.debug('h_prev: %f, h_next: %f, constraint_value: %f', h_prev, h_next, constraint_value)

        return constraint_value 

    # cost function
    def objective(u):
        # Assuming your objective is based on some mathematical operations on u
        cost = jnp.linalg.norm(u - u_nom)**2
        return cost
    
    try:
        # Step 1: Define vectorized versions of zocbf_constraint and objective
        zocbf_constraint = jax.jit(zocbf_constraint)
        objective = jax.jit(objective)

        vectorized_zocbf_constraint = jax.vmap(zocbf_constraint)
        vectorized_objective = jax.vmap(objective)

        # Step 2: Convert grid_u to a JAX array
        grid_u = jnp.array(grid_u)

        # Step 3: Compute zocbf_constraint for all u in grid_u in parallel
        zocbf_constraint_values = vectorized_zocbf_constraint(grid_u)
    

        # Step 4: Filter for safe inputs where zocbf_constraint >= 0
        safe_mask = zocbf_constraint_values >= 0
        safe_inputs = grid_u[safe_mask]

        # Step 5: Compute the cost for all safe inputs in parallel
        cost_values = vectorized_objective(safe_inputs)

        # Step 6: Find the index of the minimum cost
        min_cost_index = jnp.argmin(cost_values)

        # Step 7: Select the u corresponding to the minimum cost
        u_filtered = safe_inputs[min_cost_index]

    except Exception as e:
        logger.error('JAX implementation failed with exception: %s', e)
        u_filtered = u_nom  # nominal control if safety filter fails
        logger.warning('Filtering failed at state %s.', state)
    return u_filtered  


# sim with safety filter
def simulate_controller_rk4(controller, initial_state, Kv, Kw, grid_u,params,impl_list):
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
        filter_flag = False

        # safety filter, ZOCBF-OP
        if 'numerical_integration' in impl_list:
            control_input_filtered = filter_control_input(
                control_input, u_prev, h, params, gamma, delta, terrain_rhs, state, dt)
            filter_flag = True

        if 'parallel_simulation' in impl_list:
            control_input_filtered = filter_control_input_parsim(
                control_input, u_prev, h, params, gamma, delta, terrain_rhs, state, dt,grid_u)
            filter_flag = True
        
        if 'parallel_simulation_jax' in impl_list:
            control_input_filtered = filter_control_input_parsim_jax(
                control_input, u_prev, h_jax, params, gamma, delta, terrain_rhs_jax, state, dt,grid_u)
            filter_flag = True

        if not filter_flag:
            control_input_filtered = control_input
            logging.error('no safety filter implemention selected, nominla input applied instead. Possible options: numerical_integration, parallel_simulation,parallel_simulation_jax ')
        
        u_prev = control_input_filtered # update u_prev

        h_val = h(state, u_prev, params, dt)
        h_values.append(float(h_val)) # for plotting

        # next state with safe control input
        rhs_num = lambda t, s: terrain_rhs(t, s, control_input_filtered, params)
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
 
# Define the input bounds
bounds_min = np.array([0.0, -2.0])  # Lower bounds for each dimension
bounds_max = np.array([7.0, 2.0])    # Upper bounds for each dimension

# Define the number of points (resolution) for the grid in each dimension
grid_points_per_dim = np.array([1000, 1000])  # number of points along each dimension

# Create the grid
grid_u = create_uniform_grid(bounds_min, bounds_max, grid_points_per_dim)

# ['numerical_integration','parallel_simulation','parallel_simulation_jax']
impl_list = ['parallel_simulation_jax']

# Start the timer
start_time = timeit.default_timer()

# run sim for filtered control
(states_terrain, controls_terrain, controls_filtered, h_values) = simulate_controller_rk4(
    P_control, initial_state, Kv, Kw, grid_u, terrain_params,impl_list)

# Stop the timer
end_time = timeit.default_timer()
# Calculate and print elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# run sim for u_nom control
(states_nominal, controls_nominal, h_values_nominal) = simulate_controller_rk4_nominal(
    P_control, terrain_rhs, initial_state, Kv, Kw, terrain_params)

# profiling
# simulate_nominal_case = lambda : simulate_controller_rk4_nominal(
#     P_control, terrain_rhs, initial_state, Kv, Kw, terrain_params)
# simulate_zocbf_case = lambda : simulate_controller_rk4(
#     P_control, terrain_rhs, initial_state, Kv, Kw, terrain_params)
# cProfile.run('simulate_nominal_case()')
# cProfile.run('simulate_zocbf_case()')



print("Grid points size:\n", len(grid_u))

# Plots
set_figure_defaults()

fig_width = 4.5
fig_height = fig_width / 1.25
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
        path_style = '-'  
        if not label_unsafe_added:
            # ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2, label='$\mathbf{k_d}$, unsafe')
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2)

            label_unsafe_added = True
        else:
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2)

    else:
        path_color = 'green'
        path_style = '-'  
        if not label_safe_added:
            ax.plot(x_segment, y_segment, z_segment, linestyle=path_style, color=path_color, linewidth=2, label='$x, \mathbf{k_d}$')
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

ax.legend(loc='upper center', bbox_to_anchor=(0.1, 0.8), ncol=1, fontsize=9, handletextpad=0.2, labelspacing=0.2, borderpad=0.2)

plt.tight_layout()
plt.savefig('terrainZOCBF.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot comparison of nominal and filtered control inputs
fig_width = 5.0
fig_height = fig_width / 1.2
fig = plt.figure(figsize=(fig_width, fig_height))

plt.subplot(2, 2, 1)
plt.step(np.arange(timesteps) * dt, [c[0] for c in controls_filtered], label='$v$, ZOCBF', color='blue',)
plt.step(np.arange(timesteps) * dt, [c[0] for c in controls_nominal], label='$v, \mathbf{k_d}$', color='green')
plt.xlabel('Time [s]')
plt.ylabel('$v$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=9)

plt.subplot(2, 2, 2)
plt.step(np.arange(timesteps) * dt, [c[1] for c in controls_filtered], label='$\omega$, ZOCBF', color='blue',)
plt.step(np.arange(timesteps) * dt, [c[1] for c in controls_nominal], label='$\omega, \mathbf{k_d}$', color='green')
plt.xlabel('Time [s]')
plt.ylabel('$\omega$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=9)

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(h_values)) * dt, h_values, label='$h$, ZOCBF', color='blue', linewidth=2)
plt.plot(np.arange(len(h_values_nominal)) * dt, h_values_nominal, label='$h, \mathbf{k_d}$', color='green', linewidth=2)
plt.xticks(np.arange(0, 21, 5))  
plt.xlabel('Time [s]')
plt.ylabel('$h(u, \phi)$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig('inputs_h.pdf', format='pdf', bbox_inches='tight')
plt.show()