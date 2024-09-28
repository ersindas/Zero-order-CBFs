import sympy as sp

def double_integrator_dynamics_symbolic(x, u):
    """ Double integrator dynamics: dx/dt = f(x) + g(x) u """
    # State vector: x = [p, v] where p is position, v is velocity
    p, v = x
    # Dynamics: dp/dt = v, dv/dt = u
    f_x = sp.Matrix([v, 0])  # No control term in dp/dt
    g_x = sp.Matrix([0, 1])  # Control term only in dv/dt
    return f_x, g_x

def rk1_double_integrator_symbolic(f, x0, u, T):
    """ RK1 (Euler) for double integrator model """
    x = sp.Matrix(x0)
    f_x, g_x = f(x, u)
    k1 = f_x + g_x * u
    x_next = x + T * k1
    return x_next

def rk2_double_integrator_symbolic(f, x0, u, T):
    """ RK2 for double integrator model """
    x = sp.Matrix(x0)
    f_x, g_x = f(x, u)
    k1 = f_x + g_x * u
    f_x2, g_x2 = f(x + 0.5 * T * k1, u)
    k2 = f_x2 + g_x2 * u
    x_next = x + T * k2
    return x_next

def rk3_double_integrator_symbolic(f, x0, u, T):
    """ RK3 for double integrator model """
    x = sp.Matrix(x0)
    f_x, g_x = f(x, u)
    k1 = f_x + g_x * u
    f_x2, g_x2 = f(x + 0.5 * T * k1, u)
    k2 = f_x2 + g_x2 * u
    f_x3, g_x3 = f(x - T * k1 + 2 * T * k2, u)
    k3 = f_x3 + g_x3 * u
    x_next = x + (T / 6) * (k1 + 4 * k2 + k3)
    return x_next

def rk4_double_integrator_symbolic(f, x0, u, T):
    """ RK4 for double integrator model """
    x = sp.Matrix(x0)
    f_x, g_x = f(x, u)
    k1 = f_x + g_x * u
    f_x2, g_x2 = f(x + 0.5 * T * k1, u)
    k2 = f_x2 + g_x2 * u
    f_x3, g_x3 = f(x + 0.5 * T * k2, u)
    k3 = f_x3 + g_x3 * u
    f_x4, g_x4 = f(x + T * k3, u)
    k4 = f_x4 + g_x4 * u
    x_next = x + (T / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

def main():
    # Integration time step
    T = sp.Symbol('T')  # Time step size
    
    # Initial condition x0 = [p, v]
    p0 = sp.Symbol('p0')  # Initial position
    v0 = sp.Symbol('v0')  # Initial velocity
    x0 = [p0, v0]

    # Parametric control input u as symbolic variable
    u = sp.Symbol('u')  # Acceleration (symbolic)

    # Solve dynamics using RK1, RK2, RK3, and RK4 witT symbolic inputs
    x_next_rk1 = rk1_double_integrator_symbolic(double_integrator_dynamics_symbolic, x0, u, T)
    x_next_rk2 = rk2_double_integrator_symbolic(double_integrator_dynamics_symbolic, x0, u, T)
    x_next_rk3 = rk3_double_integrator_symbolic(double_integrator_dynamics_symbolic, x0, u, T)
    x_next_rk4 = rk4_double_integrator_symbolic(double_integrator_dynamics_symbolic, x0, u, T)

    print("RK1 (Euler) Next State:")
    sp.pprint(x_next_rk1)
    
    print("\nRK2 Next State:")
    sp.pprint(x_next_rk2)
    
    print("\nRK3 Next State:")
    sp.pprint(x_next_rk3)
    
    print("\nRK4 Next State:")
    sp.pprint(x_next_rk4)

if __name__ == "__main__":
    main()