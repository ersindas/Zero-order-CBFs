import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

T = cp.Parameter(nonneg=True)      
Gamma = cp.Parameter(nonneg=True)
Gamma2 = cp.Parameter(nonneg=True)    
delta = cp.Parameter(nonneg=True)  
u_max = cp.Parameter(nonneg=True)  
P = cp.Parameter(nonneg=True)  
u_des = cp.Parameter()    

# system dynamics wrt T
def system_dynamics(x, u):
    A = np.array([[1, T.value], [0, 1]])
    B = np.array([[T.value**2 / 2], [T.value]])  
    return A @ x.reshape(2, 1) + B * u

def h1(x):
    return P.value - x[0]

def h2(x):
    return P.value - x[0]**2

# higher-order discrete CBF for h1
def zocbf_condition_h3(x):
    v_k = x[1]  
    p_k = x[0]  
    
    phi_1 = -T.value * v_k + Gamma.value * (P.value - p_k)
    phi_2 = -T.value * v_k + Gamma.value * (P.value - p_k - T.value * v_k) - phi_1 + Gamma2.value * Gamma.value * phi_1
    
    return np.array([-T.value**2, phi_2])

# higher-order discrete CBF for h2
def zocbf_condition_h4(x):
    v_k = x[1]  
    p_k = x[0]  
    
    phi_1 = -(p_k + T.value * v_k)**2 + p_k**2 + Gamma.value * (P.value - p_k**2)
    phi_2 = -(p_k + 2 * T.value * v_k)**2 + (p_k + T.value * v_k)**2 + Gamma.value * (P.value - ( p_k + T.value * v_k)**2) - phi_1 + Gamma2.value * Gamma.value * phi_1
    
    return np.array([-T.value**4, -(2 * T.value**2) * (p_k + 2 * T.value * v_k), phi_2])

# ZOCBF conditions for h1 and h2
def zocbf_condition_h1(x):
    return np.array([-T.value**2 / 2, -T.value * x[1] - Gamma.value * x[0] + Gamma.value * P.value - delta.value])

def zocbf_condition_h2(x):
    return np.array([-T.value**4 / 4, - (T.value**2) * (x[0] + T.value * x[1]), 
                     -(x[0] + T.value * x[1])**2 + (x[0])**2 + Gamma.value * (P.value - x[0]**2) - delta.value])

# CVXPY
def setup_cvxpy_solver(x, constraint='h1'):
    u = cp.Variable()
    
    # min.  (u - u_nom)^2
    objective = cp.Minimize((u_des.value - u)**2)
    
    if constraint == 'h1':
        zocbf_h1 = zocbf_condition_h1(x)
        constraints = [zocbf_h1[0] * u >= -zocbf_h1[1], u >= -u_max.value, u <= u_max.value]
    
    elif constraint == 'h2':
        zocbf_h2 = zocbf_condition_h2(x)
        constraints = [zocbf_h2[0] * u**2 + zocbf_h2[1] * u >= -zocbf_h2[2], u >= -u_max.value, u <= u_max.value]
    
    elif constraint == 'h3':
        zocbf_h3 = zocbf_condition_h3(x)
        constraints = [zocbf_h3[0] * u >= -zocbf_h3[1], u >= -u_max.value, u <= u_max.value]

    elif constraint == 'h4':
        zocbf_h4 = zocbf_condition_h4(x)
        constraints = [zocbf_h4[0] * u**2 + zocbf_h4[1] * u >= -zocbf_h4[2], u >= -u_max.value, u <= u_max.value]

    problem = cp.Problem(objective, constraints)

    return problem, u

# simulations
def simulate_system(x0, num_steps, constraint='h1'):
    x = x0
    t = 0
    states = np.zeros((num_steps + 1, 2))
    times = np.zeros(num_steps + 1)
    controls = np.zeros(num_steps)
    
    states[0] = x
    times[0] = t
    
    for i in range(num_steps):
        problem, u_var = setup_cvxpy_solver(x, constraint=constraint)
        
        # solution
        problem.solve(solver=cp.ECOS)
        
        if problem.status != cp.OPTIMAL:
            print(f"CVXPY solver failed at step {i}. Status: {problem.status}")
            print(f"Current state: {x}")
            # u = 0  # zero if solver fails
            u = u_des.value  #nominal control input

        else:
            u = u_var.value
        
        controls[i] = u
        
        # update states
        x = system_dynamics(x, u).flatten()
        t += T.value
        
        states[i+1] = x
        times[i+1] = t
    
    return states, times, controls

# parameters
T.value = 0.1          # sampling time
Gamma.value = 1 * T.value     # gamma for the class K-function, x * Gamma = gamma 
Gamma2.value = 1 # second class-K for HOCBFs
delta.value = 0.01    # safety margin
u_max.value = 10    # minimum control input
P.value = 10          # constant in h
u_des.value = 0  # nominal contorl input

x0 = np.array([0, 2])  # x_0
num_steps = int(15 / T.value)  # 10 seconds sim

# run for h1
states_h1, times_h1, controls_h1 = simulate_system(x0, num_steps, constraint='h1')

# run for h2
states_h2, times_h2, controls_h2 = simulate_system(x0, num_steps, constraint='h2')

# run for h3 (higher-order CBF)
states_h3, times_h3, controls_h3 = simulate_system(x0, num_steps, constraint='h3')

# run for h4 (higher-order CBF)
states_h4, times_h4, controls_h4 = simulate_system(x0, num_steps, constraint='h4')

# plots
# Set the default line width and marker size
plt.rcParams['lines.linewidth'] = 2    # Default line width
plt.rcParams['lines.markersize'] = 5   # Default marker size

# Set the default axes properties
plt.rcParams['axes.linewidth'] = 2     # Default axes line width
plt.rcParams['axes.labelsize'] = 12    # Default axes font size
plt.rcParams['xtick.labelsize'] = 10   # Default x-tick font size
plt.rcParams['ytick.labelsize'] = 10   # Default y-tick font size

# Set default text properties to use LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10          # Default font size for text
plt.rcParams['legend.fontsize'] = 11    # Default legend font size
plt.rcParams['legend.title_fontsize'] = 10  # Default legend title font size

# Use LaTeX for axis tick labels and legends
plt.rcParams['axes.formatter.use_locale'] = False
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Extra LaTeX packages (if needed)



fig_width = 5.0
fig_height = fig_width / 1.3

plt.figure(figsize=(fig_width, fig_height))
# plt.set_ylim([0, 20])

plt.subplot(2, 2, 1)
plt.plot(times_h1, states_h1[:, 0], label=r'$h_1$')
plt.plot(times_h2, states_h2[:, 0], label=r'$h_2$')
plt.plot(times_h3, states_h3[:, 0], label=r'$h_1$', linestyle='--')
plt.plot(times_h4, states_h4[:, 0], label=r'$h_2$', linestyle='--')
plt.ylabel(r'$p$')
plt.legend(loc='upper right')  

plt.subplot(2, 2, 2)
plt.plot(times_h1, states_h1[:, 1], label=r'$h_1$')
plt.plot(times_h2, states_h2[:, 1], label=r'$h_2$')
plt.plot(times_h3, states_h3[:, 1], label=r'$h_1$', linestyle='--')
plt.plot(times_h4, states_h4[:, 1], label=r'$h_2$', linestyle='--')
plt.ylabel(r'$v$')
plt.legend(loc='upper right')  

plt.subplot(2, 2, 3)
plt.step(times_h1[:-1], controls_h1, label=r'$h_1$', where='post')
plt.step(times_h2[:-1], controls_h2, label=r'$h_2$', where='post')
plt.step(times_h3[:-1], controls_h3, label=r'$h_1$', where='post', linestyle='--')
plt.step(times_h4[:-1], controls_h4, label=r'$h_2$', where='post', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel(r'$u$')
plt.legend( loc='upper right')  

plt.subplot(2, 2, 4)
plt.plot(times_h1, [h1(x) for x in states_h1], label=r'$h_1$')
plt.plot(times_h2, [h2(x) for x in states_h2], label=r'$h_2$')
plt.plot(times_h3, [h1(x) for x in states_h3], label=r'$h_1$', linestyle='--') 
plt.plot(times_h4, [h2(x) for x in states_h4], label=r'$h_2$', linestyle='--')  
plt.xlabel('Time [s]')
plt.ylabel(r'$h(x)$')
plt.legend(loc='upper right')  

plt.tight_layout()
plt.savefig('h_constraints.pdf', format='pdf', bbox_inches='tight')