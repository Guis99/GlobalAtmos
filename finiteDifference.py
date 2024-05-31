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


def runShallowWaterEqWithPBC(n, IC, dt, T):
    numDof = IC.shape[0]
    numNodes = int(numDof / 3)
    nt = int(T / dt) + 1

    soln = np.zeros([nt, numDof])

    L = math.pi / 2
    dx = L / (n - 1)

    timesteps = np.linspace(0,T,num=nt)
    
    soln = odeint(shallowWaterForwardProp, IC, timesteps, args=(dx,))

    print(timesteps)

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

    ts = "impl"
    if ts == "impl":
        systemMat = getLaplacianStiffnessMatWithPBC(dx, dt, n)
        invSystemMat = np.linalg.solve(systemMat, np.eye(x.shape[0]))
    for i,t in enumerate(timesteps[1:]):
        if ts == "expl":
            x = laplaceForwardProp2ndOrder(x, dx, dt, n)
        elif ts == "impl":
            x = invSystemMat@x
        soln[i,:] = x

    print(timesteps)

    return soln

def getLaplacianStiffnessMatWithPBC(dx, dt, n):
    alpha = -dt/dx/dx
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

def shallowWaterForwardProp(y, t, dx):
    pass

def laplaceForwardProp2ndOrder(state, dx, dt, nx):
    alpha = dt/dx/dx

    out = alpha * (indexIntoState(state, 1, 0, nx) + indexIntoState(state, -1, 0, nx) + 
                indexIntoState(state, 0, 1, nx) + indexIntoState(state, 0, -1, nx))
    
    out += (1 - 4*alpha) * indexIntoState(state, 0, 0, nx)

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

def animate_surface(data):
    """
    Creates an animation of 3D surface plots from a 3D numpy array.
    
    Parameters:
    data (numpy.ndarray): 3D numpy array where the first dimension represents time
                          and the other two represent spatial dimensions x and y.
    """
    fig = plt.figure()
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

if __name__ == '__main__':
    # runShallowWaterEqWithPBC(2,np.array([1,1,1,2,2,2,3,3,3]),.1,1)
    # Is = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
    # Js = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])

    # print(sub2ind(Is, Js, 4))
    # print(np.roll(np.array([1,2,3,4]), -1))

    # print('idx2s')
    # nx = 5
    # print(indexIntoState(np.zeros(nx**2),1,1,nx))

    '''
    T is total timespan
    dt is time interval
    n is number of nodes evenly spaced across [-pi/4, pi/4]
    nt is number of timesteps, including the initial condition
    '''
    n = 11
    dx2 = (math.pi/4/(n-1))**2 / 2

    T = 1
    dt = .05
    # dt = dx2 / 1.5
    
    nt = int(T / dt) + 1

    QPI = math.pi/4

    xs = np.linspace(-QPI, QPI, n)
    ys = np.linspace(-QPI, QPI, n)

    print('test')
    print(xs, -QPI/2)
    logicx = (-QPI/2 > xs) | (xs > QPI/2)
    logicy = (-QPI/2 > ys) | (ys > QPI/2)

    [X,Y] = np.meshgrid(xs, ys)

    print('loginds')
    print(logicx,logicy)
    print('----------')
    print(X)
    print(Y)

    r2d = 180/np.pi

    IC = np.cos(4*X)*np.cos(4*Y)
    print('-----------')
    print(IC)
    IC[logicy, :] = 0
    IC[:, logicx] = 0
    
    print('--------------')
    print(IC)

    # print(xs)
    # print(ys)

    solns = runLaplaceEqWithPBC(n, IC, dt, T)
    soln_reshape = np.reshape(solns, [nt,n,n])
    print(soln_reshape[1,:,:])
    ie = np.sum(soln_reshape,axis=(1,2))
    plt.plot(ie)
    animate_surface(soln_reshape)