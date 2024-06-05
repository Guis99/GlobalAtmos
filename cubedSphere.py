import numpy as np
import math
import matplotlib.pyplot as plt

def auxVars(xi, nu):
    '''
        xi, nu : local angular coordinates in radians. [-pi/4, pi/4]
    '''
    X = np.tan(xi)
    Y = np.tan(nu)
    delta = 1 + X**2 + Y**2
    C = (1 + X**2)**.5
    D = (1 + Y**2)**.5 
    return (X, Y, delta, C, D)

def equitorialTransform(xi, nu):
    out = np.zeros((2,2))
    (X,Y,delta,C,D) = auxVars(xi, nu)
    out[1,0] = -1
    out[0,1] = C*D/(delta**.5)
    out[1,1] = X*Y/(delta**.5)

    return out

def northPoleTransform(xi, nu):
    out = np.zeros((2,2))
    (X,Y,delta,C,D) = auxVars(xi, nu)
    fac = (delta-1)**.5

    out[0,0] = D*X
    out[1,0] = C*Y
    out[0,1] = -D*Y/(delta**.5)
    out[1,1] = C*X/(delta**.5)

    out /= fac
    return out

def southPoleTransform(xi,nu):
    out = np.zeros((2,2))
    (X,Y,delta,C,D) = auxVars(xi, nu)
    fac = (delta-1)**.5

    out[0,0] = -D*X
    out[1,0] = -C*Y
    out[0,1] = D*Y/(delta**.5)
    out[1,1] = -C*X/(delta**.5)

    out /= fac
    return out

def getFaceAdjacency():
    out = np.zeros((6,4), dtype='int32')
    out[0] = [5,2,6,4]
    out[1] = [5,3,6,1]
    out[2] = [5,4,6,2]
    out[3] = [5,1,6,3]
    out[4] = [3,2,1,4]
    out[5] = [1,2,3,4]

    return out - 1

def getMetrixTensor(xi, nu, dim):
    (X,Y,delta,C,D) = auxVars(xi, nu)
    if dim==2:
        out = np.zeros(2,2)
        out[0,0] = 1
        out[1,0] = -X*Y/C/D
        out[0,1] = -X*Y/C/D
        out[1,1] = 1
    elif dim==3:
        out[0,0] = 1
        out[1,0] = -X*Y/C/D
        out[0,1] = -X*Y/C/D
        out[1,1] = 1
        out[2,2] = 1

    return out

def sphericalToLocal(phi, theta):
    '''
    Returns region as well as local angular coordinates xi and nu
    '''
    QPI = math.pi / 4
    colat = math.pi / 2 - theta

    if 0<theta<QPI:
        region = 4

    elif 3*QPI<theta<4*QPI:
        region = 5
        
    elif QPI < theta < 3*QPI:
        if -QPI < phi < QPI:
            region = 0
            xi = phi
            nu = colat
        elif QPI < phi < 3*QPI:
            region = 1
            xi = phi - 2*QPI
            nu = colat
        elif 3*QPI < phi < 5*QPI:
            region = 2
            xi = phi - 4*QPI
            nu = colat
        elif 5*QPI < phi < 7*QPI:
            region = 3
            xi = phi - 6*QPI
            nu = colat

    return (region, xi, nu)

def localToSpherical(region, xi, nu):
    pass

# grid-related

def makeGrid(N):
    width = math.pi/4
    dx = width / (N - 1)
    numNodes = N ** 2
    cubeGrids = []
    for i in range(6):
        nodes = np.arange(0,numNodes,1)
        cubeGrids.append(np.reshape(nodes, [N,N]) + i * numNodes)

    return (dx, cubeGrids)

def vizGrid(grid, dx, N):
    n = N-1
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))

    # Hide all axes, we will only show the ones with cube faces
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    # Indices for the cube faces in the axes array
    face_positions = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)]
    roman_numerals = ['V', 'IV', 'I', 'II', 'III', 'VI']

    for index, pos in enumerate(face_positions):
        ax = axes[pos[0], pos[1]]
        ax.axis('on')  # Only turn on the axes we need

        # Draw grid lines
        for i in range(n + 1):
            ax.axhline(i / n, color='black', linewidth=0.5)
            ax.axvline(i / n, color='black', linewidth=0.5)

        # Set limits and aspect
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Label each face with a Roman numeral in the center
        ax.text(0.5, 0.5, roman_numerals[index], fontsize=20, ha='center', va='center', color='red')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def sub2indOnCube(i,j,nx):
    '''i,j: numpy vectors of i and j coords'''
    return j * nx + i

def rollIndicesOnCube(offset_i, offset_j, nx):
    offset_i = -offset_i
    offset_j = -offset_j
    p = np.arange(nx)

    rolledX = np.roll(p, offset_i)
    rolledY = np.roll(p, offset_j)

    [X,Y] = np.meshgrid(rolledX, rolledY)
    iIdxs = np.reshape(X,nx*nx)
    jIdxs = np.reshape(Y,nx*nx)

    linIdx = sub2indOnCube(iIdxs, jIdxs, nx)

    return linIdx

def indexIntoStateOnCube(state, offset_i, offset_j, nx):
    linIdx = rollIndicesOnCube(offset_i, offset_j, nx)

    return state[linIdx]
    

if __name__ == '__main__':
    # adjacency = getFaceAdjacency()
    # a = np.linspace(0, 10, 11)
    # print(a)
    # print(adjacency)
    # print(type(adjacency[0]))
    # print(a[adjacency[0]])
    # print(a[adjacency[1]])
    # print(a[adjacency[2]])

    # (X,Y,delta,C,D) = auxVars(0,0)
    # print(X,Y,delta,C,D)
    vizGrid(1,1,10+1)