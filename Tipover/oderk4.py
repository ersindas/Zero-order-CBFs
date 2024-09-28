# 4th-order Runge-Kutta (RK4)
import numpy as np

def oderk4(RHS, t, x0):
    # calculation of the fixed time step
    if len(t) > 1:
        dt = (t[-1] - t[0]) / (len(t) - 1)

    # memory allocation for solution
    tsolu = t
    xsolu = np.zeros((len(t), len(x0)))

    # solution of the differential equation
    for kt in range(len(t)):
        tnew = t[kt]

        # evaluation of initial conditions
        if kt == 0:
            xnew = x0
        else:
            # RK4 stages
            k1 = dt * RHS(told, xold)
            k2 = dt * RHS(told + 0.5 * dt, xold + 0.5 * k1)
            k3 = dt * RHS(told + 0.5 * dt, xold + 0.5 * k2)
            k4 = dt * RHS(told + dt, xold + k3)

            # update solution
            xnew = xold + (k1 + 2*k2 + 2*k3 + k4) / 6
            # xnew = xold + k1


        # store solution
        xsolu[kt, :] = xnew

        # update
        told = tnew
        xold = xnew

    return tsolu, xsolu