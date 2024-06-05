from cubedSphere import *
from scipy.integrate import odeint
import matplotlib.animation as animation

'''
Node labeling convention:

x traverses fastest along inner index
y traverses fastest along outer index

data will be stored x major, row major

data is passed into solver entry pt. funcs as 2D array
data will be flattened out internally for easy manipulation, then repackaged as a 3D array of shape [nt, nx, nx]

'''

def runRossbyHaurwitzWithPBC(n, IC, dt, T, u0, alpha, a):
    nt = int(T / dt) + 1
    nNodes = n*n

    soln = np.zeros([nt, nNodes])

    L = math.pi / 2
    dx = L / (n - 1)

    timesteps = np.linspace(0,T,num=nt)

    QPI = math.pi/4
    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    [Xs,Ys] = np.meshgrid(xs, ys)

    Xs = np.reshape(Xs,[nNodes])
    Ys = np.reshape(Ys,[nNodes])

    [X,Y,delta,C,D] = auxVars(Xs, Ys)
    
    u = u0 * (C**2 * np.cos(0) + Y * np.sin(0)) / delta
    v = u0 * (X * Y * np.cos(0) - X * np.sin(0)) / delta

    u = delta * u / a / C / C
    v = delta * v / a / D / D

    r11 = np.cos(alpha)
    r12 = -np.sin(alpha)
    r21 = np.sin(alpha)
    r22 = np.cos(alpha)

    u_rot = r11 * u + r12 * v
    v_rot = r21 * u + r22 * v

    c2=plt.quiver(Xs, Ys, u_rot, v_rot, angles='xy', scale_units='xy', scale=80)

    IC_rm = np.reshape(IC, n*n)
    
    ts = "expl"
    if ts == "expl":
        soln = odeint(rossbyHaurwitzForwardProp, IC_rm, timesteps, args=(dx,n,u_rot,v_rot))
    elif ts == "impl":
        pass

    print(soln)

    return soln

def runShallowWaterEqWithPBC(n, IC, dt, T):
    numDof = IC.shape[0]
    numNodes = int(numDof / 3)
    nt = int(T / dt) + 1

    soln = np.zeros([nt, numDof])

    L = math.pi / 2
    dx = L / (n - 1)

    timesteps = np.linspace(0,T,num=nt)
    
    ts = "expl"
    if ts == "expl":
        soln = odeint(shallowWaterForwardProp, IC, timesteps, args=(dx,numNodes))
    elif ts == "impl":
        pass

    return soln

def runLaplaceEqWithPBC(n, IC, dt, T):
    IC_rm = np.reshape(IC, n*n)
    numDof = IC_rm.shape[0]
    nt = int(T / dt) + 1

    soln = np.zeros([nt, numDof])
    soln[0,:] = IC_rm

    L = math.pi / 2
    dx = L / (n - 1)

    timesteps = np.linspace(0,T,num=nt)
    x = IC_rm

    ts = "expl_int"
    if ts == "expl_int":
        systemMat = getLaplacianStiffnessMatWithPBC(-dx, 1, n, isforward=True)
        soln = odeint(laplaceForwardPropOdeInt, IC_rm, timesteps, args=(dx,n,))
    else:
        if ts == "impl":
            systemMat = getLaplacianStiffnessMatWithPBC(dx, dt, n)
            invSystemMat = np.linalg.solve(systemMat, np.eye(x.shape[0]))
        for i,t in enumerate(timesteps[1:]):
            if ts == "expl":
                x = laplaceForwardProp2ndOrder(x, dx, dt, n)
            elif ts == "impl":
                x = invSystemMat@x
            soln[i,:] = x

    return soln

def runLaplaceEqOnCube(n, IC, dt, T):
    IC_rm = np.reshape(IC, n*n)
    numDof = IC_rm.shape[0]
    nt = int(T / dt) + 1

    soln = np.zeros([nt, numDof])
    soln[0,:] = IC_rm

    L = math.pi / 2
    dx = L / (n - 1)

    timesteps = np.linspace(0,T,num=nt)
    x = IC_rm

    a = 1
    QPI = math.pi/4
    nNodes = n * n
    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    [Xs,Ys] = np.meshgrid(xs, ys)

    Xs = np.reshape(Xs,[nNodes])
    Ys = np.reshape(Ys,[nNodes])

    [X,Y,delta,C,D] = auxVars(Xs, Ys)

    ts = "expl_int"
    if ts == "expl_int":
        # soln = odeint(laplaceForwardPropOnCubeOdeInt, IC_rm, timesteps, args=(dx,n,))
        soln = odeint(laplaceForwardPropOdeIntCubeGrid, IC_rm, timesteps, args=(dx, n, X, Y, delta, C, D, a))
    else:
        if ts == "impl":
            systemMat = getLaplacianStiffnessMatWithPBC(dx, dt, n)
            invSystemMat = np.linalg.solve(systemMat, np.eye(x.shape[0]))
        for i,t in enumerate(timesteps[1:]):
            if ts == "expl":
                # x = laplaceForwardProp2ndOrder(x, dx, dt, n)
                pass
            elif ts == "impl":
                x = invSystemMat@x
            soln[i,:] = x

    return soln

def laplaceForwardPropOnCubeOdeInt(y, t, dx, nx):
    alpha = 1/dx/dx

    out = alpha * (indexIntoStateOnCube(y, 1, 0, nx) + indexIntoStateOnCube(y, -1, 0, nx) + 
                indexIntoStateOnCube(y, 0, 1, nx) + indexIntoStateOnCube(y, 0, -1, nx) - 4*indexIntoStateOnCube(y, 0, 0, nx))

    return out

def rossbyHaurwitzForwardProp(y, t, dx, n, u, v):
    div = 12 * dx

    xp1 = indexIntoState(y, 1, 0, n)
    xm1 = indexIntoState(y, -1, 0, n)

    xp2 = indexIntoState(y, 2, 0, n)
    xm2 = indexIntoState(y, -2, 0, n)

    yp1 = indexIntoState(y, 0, 1, n)
    ym1 = indexIntoState(y, 0, -1, n)

    yp2 = indexIntoState(y, 0, 2, n)
    ym2 = indexIntoState(y, 0, -2, n)

    # linear advection
    dhdx = (-xp2 + 8 * xp1 - 8 * xm1 + xm2) / div
    dhdy = (-yp2 + 8 * yp1 - 8 * ym1 + ym2) / div

    # biharmonic operator


    out = -u * dhdx - v * dhdy

    return out

def shallowWaterForwardProp(y, t, dx, numNodes):
    offset1 = numNodes
    offset2 = 2*numNodes

    u = y[:offset1]
    v = y[offset1:offset2]
    h = y[offset2:]

# TODO
    dudt = u
    dvdt = v
    dhdt = h

    out = np.hstack([dudt,dvdt,dhdt])

    return out

def getLaplacianStiffnessMatWithPBC(dx, dt, n, isforward=False):
    alpha = -dt/dx/dx
    if isforward:
        center = -4*alpha
    else:
        center = 1 - 4*alpha

    systemSize = n*n
    out = np.zeros([systemSize,systemSize])

    cols = np.arange(n*n)

    row1 = rollIndices(1,0,n)
    row2 = rollIndices(-1,0,n)
    row3 = rollIndices(0,1,n)
    row4 = rollIndices(0,-1,n)

    out[cols, cols] = center
    out[cols, row1] = alpha
    out[cols, row2] = alpha
    out[cols, row3] = alpha
    out[cols, row4] = alpha 

    return out

def laplaceForwardProp2ndOrder(state, dx, dt, nx):
    alpha = dt/dx/dx

    out = alpha * (indexIntoState(state, 1, 0, nx) + indexIntoState(state, -1, 0, nx) + 
                indexIntoState(state, 0, 1, nx) + indexIntoState(state, 0, -1, nx))
    
    out += (1 - 4*alpha) * indexIntoState(state, 0, 0, nx)

    return out

def laplaceForwardPropOdeInt(y, t, dx, nx):
    alpha = 1/dx/dx

    out = alpha * (indexIntoState(y, 1, 0, nx) + indexIntoState(y, -1, 0, nx) + 
                indexIntoState(y, 0, 1, nx) + indexIntoState(y, 0, -1, nx) - 4*indexIntoState(y, 0, 0, nx))

    # out = mat@y

    return out

def laplaceForwardPropOdeIntCubeGrid(y, t, dx, nx, X, Y, delta, C, D, a):
    alpha = 1/dx/dx

    term1 = alpha * (indexIntoState(y, 1, 0, nx) - 2*indexIntoState(y, 0, 0, nx) + indexIntoState(y, -1, 0, nx)) / C / C
    term2 = alpha * (indexIntoState(y, 0, 1, nx) - 2*indexIntoState(y, 0, 0, nx) + indexIntoState(y, 0, -1, nx)) / D / D
    term3 = 2 * alpha * X * Y * (indexIntoState(y, 1, 1, nx) - indexIntoState(y, -1, 1, nx) - 
                                 indexIntoState(y, 1, -1, nx) + indexIntoState(y, -1, -1, nx)) / C / C / D / D / 4

    out = delta * (term1 + term2 + term3) / a / a

    # out = mat@y

    return out

def sub2ind(i,j,nx):
    '''i,j: numpy vectors of i and j coords'''
    return j * nx + i

def rollIndices(offset_i, offset_j, nx):
    offset_i = -offset_i
    offset_j = -offset_j
    p = np.arange(nx)

    rolledX = np.roll(p, offset_i)
    rolledY = np.roll(p, offset_j)

    [X,Y] = np.meshgrid(rolledX, rolledY)
    iIdxs = np.reshape(X,nx*nx)
    jIdxs = np.reshape(Y,nx*nx)

    linIdx = sub2ind(iIdxs, jIdxs, nx)

    return linIdx

def indexIntoState(state, offset_i, offset_j, nx):
    linIdx = rollIndices(offset_i, offset_j, nx)

    return state[linIdx]

def animate_surface(data, data2):
    """
    Creates an animation of 3D surface plots from a 3D numpy array.
    
    Parameters:
    data (numpy.ndarray): 3D numpy array where the first dimension represents time
                          and the other two represent spatial dimensions x and y.
    """
    fig = plt.figure(1)
    fig2 = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[2])
    x, y = np.meshgrid(x, y)
    
    def update_frame(i):
        ax.clear()
        ax.plot_surface(x, y, data[i, :, :], cmap='viridis')
        ax.set_zlim(np.min(data), np.max(data))
        return ax
    
    ani = animation.FuncAnimation(fig, update_frame, frames=data.shape[0], repeat=True)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Surface Plot Animation')
    plt.show()

    ax = fig2.add_subplot(111, projection='3d')
    
    x = np.arange(data2.shape[1])
    y = np.arange(data2.shape[2])
    x, y = np.meshgrid(x, y)
    
    def update_frame(i):
        ax.clear()
        ax.plot_surface(x, y, data2[i, :, :], cmap='viridis')
        ax.set_zlim(np.min(data2), np.max(data2))
        return ax
    
    ani = animation.FuncAnimation(fig2, update_frame, frames=data2.shape[0], repeat=True)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Surface Plot Animation')
    plt.show()


def vizGrid(data):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))

    # Hide all axes, we will only show the ones with cube faces
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    # Indices for the cube faces in the axes array
    face_positions = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)]
    roman_numerals = ['V', 'IV', 'I', 'II', 'III', 'VI']

    x = np.arange(data.shape[1])
    y = np.arange(data.shape[2])
    x, y = np.meshgrid(x, y)
    
    def update_frame(i):
        ax.clear()
        contour = ax.contourf(data[i, :, :], cmap='viridis')
        return contour.collections

    for index, pos in enumerate(face_positions):
        ax = axes[pos[0], pos[1]]
        ax.axis('on')  # Only turn on the axes we need

        ani = animation.FuncAnimation(fig, update_frame, frames=data.shape[0], repeat=True)

        # Label each face with a Roman numeral in the center
        ax.text(0.5, 0.5, roman_numerals[index], fontsize=20, ha='center', va='center', color='red')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()

def animate_cube_faces(data):
    """
    Creates an animation of contour plots from a 3D numpy array on the 6 faces of a cube.
    
    Parameters:
    data (numpy.ndarray): 3D numpy array where the first dimension represents time
                          and the other two represent spatial dimensions x and y.
    """
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))

    # Hide all axes, we will only show the ones with cube faces
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    # Indices for the cube faces in the axes array
    face_positions = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)]
    roman_numerals = ['V', 'IV', 'I', 'II', 'III', 'VI']

    def update_frame(i):
        for index, pos in enumerate(face_positions):
            ax = axes[pos[0], pos[1]]
            ax.clear()
            ax.contourf(data[index][i,:,:], cmap='viridis')
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, data[index].shape[1])
            ax.set_ylim(0, data[index].shape[2])
            ax.set_aspect('equal')

            # Label each face with a Roman numeral in the center
            ax.text(data[index].shape[1] / 2, data[index].shape[2] / 2, roman_numerals[index],
                    fontsize=20, ha='center', va='center', color='red')
        
        # Adjust the spacing to make sure the squares touch each other
        plt.subplots_adjust(wspace=0, hspace=0)
        return axes.flatten()

    ani = animation.FuncAnimation(fig, update_frame, frames=data[0].shape[0], repeat=True)
    
    plt.show()

def cosineBell(n):
    QPI = math.pi/4

    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    logicx = (-QPI/2 > xs) | (xs > QPI/2)
    logicy = (-QPI/2 > ys) | (ys > QPI/2)

    [X,Y] = np.meshgrid(xs, ys)

    IC = np.cos(4*X)*np.cos(4*Y)

    IC[logicy, :] = 0
    IC[:, logicx] = 0

    return IC

def driver_laplace():
    n = 31

    T = 1
    dt = .05
    # dt = dx2 / 1.5
    
    nt = int(T / dt) + 1

    QPI = math.pi/4

    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    [X,Y] = np.meshgrid(xs, ys)

    IC = cosineBell(n)

    solns = runLaplaceEqWithPBC(n, IC, dt, T)
    soln_reshape = np.reshape(solns, [nt,n,n])
    ie = np.sum(soln_reshape,axis=(1,2))
    plt.plot(ie) 
    animate_surface(soln_reshape)

def driver_rossby_haurwitz():
    n = 31

    T = 1
    dt = .05
    # dt = dx2 / 1.5
    
    nt = int(T / dt) + 1

    QPI = math.pi/4

    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    [X,Y] = np.meshgrid(xs, ys)

    IC = cosineBell(n)

    alpha = .5*math.pi/2
    a = 1
    u0 = 2*math.pi*a / T

    solns = runRossbyHaurwitzWithPBC(n, IC, dt, T, u0, alpha, a)
    soln_reshape = np.reshape(solns, [nt,n,n])
    # ie = np.sum(soln_reshape,axis=(1,2))
    # plt.plot(ie) 
    animate_contour(soln_reshape)


def driver_laplace_cube():
    n = 31

    T = 1
    dt = .05
    # dt = dx2 / 1.5
    
    nt = int(T / dt) + 1

    QPI = math.pi/4

    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    [X,Y] = np.meshgrid(xs, ys)

    IC = cosineBell(n)

    solns = runLaplaceEqOnCube(n, IC, dt, T)
    soln_reshape = np.reshape(solns, [nt,n,n])

    solns_cart = runLaplaceEqWithPBC(n, IC, dt, T)
    soln_reshape_cart = np.reshape(solns_cart, [nt,n,n])

    soln_reshape_4d = list()

    soln_reshape_4d.append(soln_reshape[::-1,:,:])
    soln_reshape_4d.append(soln_reshape+20)
    soln_reshape_4d.append(soln_reshape+30)
    soln_reshape_4d.append(soln_reshape)
    soln_reshape_4d.append(np.sin(soln_reshape))
    soln_reshape_4d.append(10*soln_reshape)

    # animate_cube_faces(soln_reshape_4d)
    animate_surface(soln_reshape, soln_reshape_cart)

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


if __name__ == '__main__':
    '''
    T is total timespan
    dt is time interval
    n is number of nodes evenly spaced across [-pi/4, pi/4]
    nt is number of timesteps, including the initial condition
    '''
    # driver_laplace()
    # driver_rossby_haurwitz()
    driver_laplace_cube()