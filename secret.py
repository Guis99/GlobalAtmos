import numpy as np
from scipy.integrate import odeint

import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# utils

def sub2ind(i,j,nx):
    '''i,j: numpy vectors of i and j coords'''
    return j * nx + i

def rollIndicesToBC(offset_i, offset_j, nx):
    p = np.arange(nx)

    rolledX = p + offset_i
    rolledY = p + offset_j

    [X,Y] = np.meshgrid(rolledX, rolledY)
    iIdxs = np.reshape(X,nx*nx)
    jIdxs = np.reshape(Y,nx*nx)

    linIdx = sub2ind(iIdxs, jIdxs, nx)
    linIdx[linIdx>nx*nx] = nx*nx
    linIdx[linIdx<0] = nx*nx 

    return linIdx

def indexIntoState(state, offset_i, offset_j, nx):
    linIdx = rollIndicesToBC(offset_i, offset_j, nx)
    bc = 1

    # print(state.shape)
    state = np.hstack([state, bc])

    # print(linIdx.shape)
    # print(state.shape)

    return state[linIdx]

def coupledDiffusionReactionForwardPropOdeInt(y, t, dx, nx, kappa, s1, s2):
    alpha = kappa / dx / dx
    offset1 = (y.shape[0]) // 2
    offset2 = y.shape[0]

    # print('OFFSETS', offset1, offset2)

    p = y[:offset1]
    c = y[offset1:offset2]

    term1 = alpha * (indexIntoState(p, 1, 0, nx) + indexIntoState(p, -1, 0, nx) + 
                indexIntoState(p, 0, 1, nx) + indexIntoState(p, 0, -1, nx) - 4 * p)
    term2 = s1 * p * c

    dpdt = term1 - term2
    dcdt = -s2 * p * c

    out = np.hstack([dpdt, dcdt])
    return out

def coupledDiffusionReactionStiffnessMat(dx, nx, dt, kappa, s1, s2):
    alpha = kappa / dx / dx
    size1 = nx * nx
    size2 = 2 * size1

    out = np.zeros([size2,size2])

    cols = np.arange(size2)

def runDiffReacWithFixedBC(nx, IC, dt, T, L, kappa, s1, s2):
    numDof = IC.shape[0]
    nt = int(T / dt) + 1

    soln = np.zeros([nt, numDof])
    soln[0,:] = IC

    dx = L / (nx - 1)

    timesteps = np.linspace(0,T,num=nt)
    x = IC

    ts = "expl_int"
    if ts == "expl_int":
        soln = odeint(coupledDiffusionReactionForwardPropOdeInt, IC, timesteps, args=(dx, nx, kappa, s1, s2))
    elif ts == "impl":
        systemMat = coupledDiffusionReactionStiffnessMat(dx, nx, dt, kappa, s1, s2)
        invSystemMat = np.linalg.solve(systemMat, np.eye(x.shape[0]))
        for i,t in enumerate(timesteps[1:]):
            x = invSystemMat@x
            soln[i,:] = x

    return soln

def animate_contour(data):
    """
    Creates an animation of contour plots from a 3D numpy array.
    
    Parameters:
    data (numpy.ndarray): 3D numpy array where the first dimension represents time
                          and the other two represent spatial dimensions x and y.
    """
    fig, ax = plt.subplots()
    
    def update_frame(i):
        ax.clear()
        contour = ax.contourf(data[i, :, :], cmap='viridis')
        return contour.collections

    ani = animation.FuncAnimation(fig, update_frame, frames=data.shape[0], repeat=True)
    
    plt.colorbar(ax.contourf(data[0, :, :], cmap='viridis'), ax=ax)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour Plot Animation')
    plt.show()



def buildIC(locs, radii, L, dx, nx):
    nNodes = nx * nx
    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, L, nx)

    [Xs,Ys] = np.meshgrid(xs, ys)

    Xs = np.reshape(Xs,[nNodes])
    Ys = np.reshape(Ys,[nNodes])

    ic_size = len(locs)

    ic1 = np.ones(nNodes)
    ic2 = np.zeros(nNodes)

    for i in range(ic_size):
        xc, yc = locs[i]
        xdiff = Xs - xc
        ydiff = Ys - yc

        dists = np.sqrt(xdiff**2 + ydiff**2)
        distmask = dists < radii[i]
        ic1[distmask] = 0
        ic2[distmask] = 1

    ic = np.hstack([ic1,ic2])
    return ic

def driver_diffusion_reaction():
    L = 15
    T = 2
    dt = .05
    nx = 51
    dx = L / (nx - 1)
    nt = int(T / dt) + 1
    
    kappa = 1
    s1 = 1
    s2 = 10

    locs = [(2,2), (2,8), (5,4), (9,9),(12,3)]
    radii = [2, 1, .5, 1,3]
    IC = buildIC(locs, radii, L, dx, nx)

    soln = runDiffReacWithFixedBC(nx, IC, dt, T, L, kappa, s1, s2)
    print(soln.shape)
    soln_cells = soln[:,nx*nx:2*nx*nx]
    soln_reshape = np.reshape(soln_cells, [nt,nx,nx])
    
    animate_contour(soln_reshape)

if __name__ == '__main__':
    driver_diffusion_reaction() 




