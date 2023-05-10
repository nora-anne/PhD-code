import numpy as np
import rebound
from rebound import hash as h

def rotate_to_IP_angles(M,P,e,I,om=np.nan):
    """
    Rotates system into invariable plane.
    Inputs are vectors of mass, period, ecc, inc, M[0] is Mstar.
    Optional input of omega, if none, randomized.
    All in same units, angles in radians.
    Outputs planets' inc, Omega, omega (astrocentric).
    """

    if isinstance(om,float):
        om = np.full([len(P)],np.nan)
    sim = rebound.Simulation()
    sim.add(m=M[0])
    for i in range(1,len(M)):
        if np.isnan(om[i-1]):
            sim.add(m=M[i],P=P[i-1],e=e[i-1],inc=I[i-1],
                    omega='uniform',hash=i)
        else:
            sim.add(m=M[i],P=P[i-1],e=e[i-1],inc=I[i-1],
                    omega=om[i-1],hash=i)
    
    p = sim.particles
    sim.move_to_com()

    #initial particle coordinates, Cartesian
    pos_xyz1 = np.empty((3,sim.N))
    vel_xyz1 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz1[0,j] = p[h(j)].x
        pos_xyz1[1,j] = p[h(j)].y
        pos_xyz1[2,j] = p[h(j)].z
        vel_xyz1[0,j] = p[h(j)].vx
        vel_xyz1[1,j] = p[h(j)].vy
        vel_xyz1[2,j] = p[h(j)].vz

    #calculate ang mom vector
    L = sim.calculate_angular_momentum()
    Lmag = np.linalg.norm(L)
    Lhat = L/Lmag
##    print('initial Lhat',Lhat)
    if abs(Lhat[0]) < 1e-4 and abs(Lhat[1]) < 1e-4 and abs(1-Lhat[2]) < 1e-4:
        print('initial system in IP')
        return np.array([I,np.zeros(sim.N-1),om])

    #calculate angle to rotate around z axis
    zhat = np.array([0,0,1])
    xhat = np.array([1,0,0])
    nhat = np.cross(Lhat,zhat)
    cosOm = np.dot(nhat,xhat)
    sinOm = np.linalg.norm(np.cross(nhat,xhat))
    #z axis rotational matrix
    Rz = np.array([[cosOm,-sinOm,0],[sinOm,cosOm,0],[0,0,1]])

    #calculate new pos/vel vectors
    pos_xyz2 = np.empty((3,sim.N))
    vel_xyz2 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz2[:,j] = np.matmul(Rz,pos_xyz1[:,j])
        vel_xyz2[:,j] = np.matmul(Rz,vel_xyz1[:,j])

    #calculate angle to rotate around new x axis
    coszang = np.dot(Lhat,zhat)
    sinzang = np.linalg.norm(np.cross(Lhat,zhat))
    #x axis rotational matrix
    Rx = np.array([[1,0,0],[0,coszang,-sinzang],[0,sinzang,coszang]])

    #calculate new pos/vel vectors
    pos_xyz = np.empty((3,sim.N))
    vel_xyz = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz[:,j] = np.matmul(Rx,pos_xyz2[:,j])
        vel_xyz[:,j] = np.matmul(Rx,vel_xyz2[:,j])

    #update particle coordinates in simulation
    for j in range(sim.N):
        p[h(j)].xyz = pos_xyz[:,j]
        p[h(j)].vxyz = vel_xyz[:,j]

    L2 = sim.calculate_angular_momentum()
    L2mag = np.linalg.norm(L2)
    Lhat2 = L2/L2mag
    if abs(Lhat2[0]) > 1e-4 or abs(Lhat2[1]) > 1e-4 or abs(1-Lhat2[2]) > 1e-4:
        print('OOPS, rotated Lhat',Lhat2)

    output = np.empty((3,sim.N-1))

    o = sim.calculate_orbits(primary=p[0])
    for j in range(sim.N-1):
        output[0,j] = o[j].inc
        output[1,j] = o[j].Omega
        output[2,j] = o[j].omega
        
    return output

def rotate_to_IP_sim(M,P,e,I,om=np.nan):
    """
    Rotates system into invariable plane.
    Inputs are vectors of mass, period, ecc, inc, M[0] is Mstar.
    Optional input of omega, if none, randomized.
    All in same units, angles in radians.
    Outputs simulation object.
    """

    if isinstance(om,float):
        om = np.full([len(P)],np.nan)
    sim = rebound.Simulation()
    sim.units = ['day','Mearth','AU']
    sim.add(m=M[0])
    for i in range(1,len(M)):
        if np.isnan(om[i-1]):
            sim.add(m=M[i],P=P[i-1],e=e[i-1],inc=I[i-1],
                    omega='uniform',hash=i)
        else:
            sim.add(m=M[i],P=P[i-1],e=e[i-1],inc=I[i-1],
                    omega=om[i-1],hash=i)
    
    p = sim.particles
    sim.move_to_com()

    #initial particle coordinates, Cartesian
    pos_xyz1 = np.empty((3,sim.N))
    vel_xyz1 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz1[0,j] = p[h(j)].x
        pos_xyz1[1,j] = p[h(j)].y
        pos_xyz1[2,j] = p[h(j)].z
        vel_xyz1[0,j] = p[h(j)].vx
        vel_xyz1[1,j] = p[h(j)].vy
        vel_xyz1[2,j] = p[h(j)].vz

    #calculate ang mom vector
    L = sim.calculate_angular_momentum()
    Lmag = np.linalg.norm(L)
    Lhat = L/Lmag
    #print('initial Lhat',Lhat)

    #calculate angle to rotate around z axis
    zhat = np.array([0,0,1])
    xhat = np.array([1,0,0])
    nhat = np.cross(Lhat,zhat)
    cosOm = np.dot(nhat,xhat)
    sinOm = np.linalg.norm(np.cross(nhat,xhat))
    #z axis rotational matrix
    Rz = np.array([[cosOm,-sinOm,0],[sinOm,cosOm,0],[0,0,1]])

    #calculate new pos/vel vectors
    pos_xyz2 = np.empty((3,sim.N))
    vel_xyz2 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz2[:,j] = np.matmul(Rz,pos_xyz1[:,j])
        vel_xyz2[:,j] = np.matmul(Rz,vel_xyz1[:,j])

    #calculate angle to rotate around new x axis
    coszang = np.dot(Lhat,zhat)
    sinzang = np.linalg.norm(np.cross(Lhat,zhat))
    #x axis rotational matrix
    Rx = np.array([[1,0,0],[0,coszang,-sinzang],[0,sinzang,coszang]])

    #calculate new pos/vel vectors
    pos_xyz = np.empty((3,sim.N))
    vel_xyz = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz[:,j] = np.matmul(Rx,pos_xyz2[:,j])
        vel_xyz[:,j] = np.matmul(Rx,vel_xyz2[:,j])

    #update particle coordinates in simulation
    for j in range(sim.N):
        p[h(j)].xyz = pos_xyz[:,j]
        p[h(j)].vxyz = vel_xyz[:,j]

    L2 = sim.calculate_angular_momentum()
    L2mag = np.linalg.norm(L2)
    Lhat2 = L2/L2mag
    if abs(Lhat2[0]) > 1e-4 or abs(Lhat2[1]) > 1e-4 or abs(1-Lhat2[2]) > 1e-4:
        print('OOPS, rotated Lhat',Lhat2)

    return sim

def rotate_to_IP_sim_cartesian(M,xyz,vxyz):
    """
    Rotates system into invariable plane.
    Inputs are masses and vectors of position and velocity (N,Nx3).
    Outputs simulation object.
    """

    N = len(M)
    sim = rebound.Simulation()
    sim.units = ['day','Mearth','AU']
    for i in range(N):
        sim.add(m=M[i],x=xyz[i,0],y=xyz[i,1],z=xyz[i,2],
                vx=vxyz[i,0],vy=vxyz[i,1],vz=vxyz[i,2],hash=i)
    p = sim.particles
    sim.move_to_com()

    #initial particle coordinates, Cartesian
    pos_xyz1 = np.empty((3,sim.N))
    vel_xyz1 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz1[0,j] = p[h(j)].x
        pos_xyz1[1,j] = p[h(j)].y
        pos_xyz1[2,j] = p[h(j)].z
        vel_xyz1[0,j] = p[h(j)].vx
        vel_xyz1[1,j] = p[h(j)].vy
        vel_xyz1[2,j] = p[h(j)].vz

    #calculate ang mom vector
    L = sim.calculate_angular_momentum()
    Lmag = np.linalg.norm(L)
    Lhat = L/Lmag
    #print('initial Lhat',Lhat)

    #calculate angle to rotate around z axis
    zhat = np.array([0,0,1])
    xhat = np.array([1,0,0])
    nhat = np.cross(Lhat,zhat)
    cosOm = np.dot(nhat,xhat)
    sinOm = np.linalg.norm(np.cross(nhat,xhat))
    #z axis rotational matrix
    Rz = np.array([[cosOm,-sinOm,0],[sinOm,cosOm,0],[0,0,1]])

    #calculate new pos/vel vectors
    pos_xyz2 = np.empty((3,sim.N))
    vel_xyz2 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz2[:,j] = np.matmul(Rz,pos_xyz1[:,j])
        vel_xyz2[:,j] = np.matmul(Rz,vel_xyz1[:,j])

    #calculate angle to rotate around new x axis
    coszang = np.dot(Lhat,zhat)
    sinzang = np.linalg.norm(np.cross(Lhat,zhat))
    #x axis rotational matrix
    Rx = np.array([[1,0,0],[0,coszang,-sinzang],[0,sinzang,coszang]])

    #calculate new pos/vel vectors
    pos_xyz = np.empty((3,sim.N))
    vel_xyz = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz[:,j] = np.matmul(Rx,pos_xyz2[:,j])
        vel_xyz[:,j] = np.matmul(Rx,vel_xyz2[:,j])

    #update particle coordinates in simulation
    for j in range(sim.N):
        p[h(j)].xyz = pos_xyz[:,j]
        p[h(j)].vxyz = vel_xyz[:,j]

    L2 = sim.calculate_angular_momentum()
    L2mag = np.linalg.norm(L2)
    Lhat2 = L2/L2mag
    if abs(Lhat2[0]) > 1e-4 or abs(Lhat2[1]) > 1e-4 or abs(1-Lhat2[2]) > 1e-4:
        print('OOPS, rotated Lhat',Lhat2)

    return sim

def rotate_to_IP_sim_sim(input_sim):
    """
    Rotates system into invariable plane.
    Input is simulation object.
    Outputs simulation object.
    """

    sim = input_sim.copy()
    N = sim.N
    p = sim.particles
    sim.move_to_com()

    #initial particle coordinates, Cartesian
    pos_xyz1 = np.empty((3,sim.N))
    vel_xyz1 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz1[0,j] = p[j].x
        pos_xyz1[1,j] = p[j].y
        pos_xyz1[2,j] = p[j].z
        vel_xyz1[0,j] = p[j].vx
        vel_xyz1[1,j] = p[j].vy
        vel_xyz1[2,j] = p[j].vz

    #calculate ang mom vector
    L = sim.calculate_angular_momentum()
    Lmag = np.linalg.norm(L)
    Lhat = L/Lmag
    if abs(Lhat[0]) <= 1e-4 and\
       abs(Lhat[1]) <= 1e-4 and\
       abs(1-Lhat[2]) <= 1e-4: #if already in IP, returns original sim
        return True,sim
    #print('initial Lhat',Lhat)

    #calculate angle to rotate around z axis
    zhat = np.array([0,0,1])
    xhat = np.array([1,0,0])
    nhat = np.cross(Lhat,zhat)
    cosOm = np.dot(nhat,xhat)
    sinOm = np.linalg.norm(np.cross(nhat,xhat))
    #z axis rotational matrix
    Rz = np.array([[cosOm,-sinOm,0],[sinOm,cosOm,0],[0,0,1]])

    #calculate new pos/vel vectors
    pos_xyz2 = np.empty((3,sim.N))
    vel_xyz2 = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz2[:,j] = np.matmul(Rz,pos_xyz1[:,j])
        vel_xyz2[:,j] = np.matmul(Rz,vel_xyz1[:,j])

    #calculate angle to rotate around new x axis
    coszang = np.dot(Lhat,zhat)
    sinzang = np.linalg.norm(np.cross(Lhat,zhat))
    #x axis rotational matrix
    Rx = np.array([[1,0,0],[0,coszang,-sinzang],[0,sinzang,coszang]])

    #calculate new pos/vel vectors
    pos_xyz = np.empty((3,sim.N))
    vel_xyz = np.empty((3,sim.N))
    for j in range(sim.N):
        pos_xyz[:,j] = np.matmul(Rx,pos_xyz2[:,j])
        vel_xyz[:,j] = np.matmul(Rx,vel_xyz2[:,j])

    #update particle coordinates in simulation
    for j in range(sim.N):
        p[j].xyz = pos_xyz[:,j]
        p[j].vxyz = vel_xyz[:,j]

    L2 = sim.calculate_angular_momentum()
    L2mag = np.linalg.norm(L2)
    Lhat2 = L2/L2mag
    if abs(Lhat2[0]) > 1e-4 or abs(Lhat2[1]) > 1e-4 or abs(1-Lhat2[2]) > 1e-4:
##        print('OOPS, rotated Lhat',Lhat2)
        return False,sim

    return True,sim
