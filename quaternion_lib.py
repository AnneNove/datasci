###quaternion_lib

import numpy as np
import scipy as sp

def axisAngle2quatern(axis,angle):
    '''
    converts an axis-angle rotation to a quaternion where a 3D rotation
    is described by an anglular rotation around axis defined by a vector
    '''
    q = [0,0,0,0]
    q[0] = np.cos(angle/2)
    q[1] = -axis[0]*np.sin(angle/2)
    q[2] = -axis[1]*np.sin(angle/2)
    q[3] = -axis[2]*np.sin(angle/2)

    return q


def axisAngle2rotMat(axis,angle):
    '''
    converts an axis-angle orientation to a rotation matrix where a 3D
    rotation is described by an angular rotation around axis defined by a
    vector.
    '''
    kx = axis[0]
    ky = axis[1]
    kz = axis[2]
    cT = np.cos(angle)
    sT = np.sin(angle)
    vT = 1 - np.cos(angle)

    R = [[0,0,0],[0,0,0],[0,0,0]]

    R[0][0] = kx*kx*vT + cT
    R[0][1] = kx*ky*vT - kz*sT
    R[0][2] = kx*kz*vT + ky*sT
    
    R[1][0] = kx*ky*vT + kz*sT
    R[1][1] = ky*ky*vT + cT
    R[1][2] = ky*kz*vT - kx*sT
    
    R[2][0] = kx*kz*vT - ky*sT
    R[2][1] = ky*kz*vT + kx*sT
    R[2][2] = kz*kz*vT +cT
    
    return R

def euler2rotMat(phi,theta,psi):
    '''
    converts ZYX euler angle orientation to a rotation matrix where phi is
    a rotation around X, theta around Y and psi around Z.
    '''
    R = [[0,0,0],[0,0,0],[0,0,0]]

    R[0][0] = np.cos(psi)*np.cos(theta)
    R[0][1] = -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi)
    R[0][2] = np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)
    
    R[1][0] = np.sin(psi)*np.cos(theta)
    R[1][1] = np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi)
    R[1][2] = -np.cos(psi)*np.sin(phi) + np.sin(psi)*np.sin(theta)*np.cos(phi)
    
    R[2][0] = -np.sin(theta)
    R[2][1] = np.cos(theta)*np.sin(phi)
    R[2][2] = np.cos(theta)*np.cos(phi)
    
    return R


def quatern2euler(q):
    '''
    converts a quaternion orientation ZYX euler angles
    '''

    R = [[0,0,0],[0,0,0],[0,0,0]]

    R[0][0] = 2*q[0]**2-1+2*q[1]**2
    R[1][0] = 2*(q[1]*q[2]-q[0]*q[3])
    R[2][0] = 2*(q[1]*q[3]+q[0]*q[2])
    R[2][1] = 2*(q[2]*q[3]-q[0]*q[1])
    R[2][2] = 2*q[0]**2-1+2*q[3]**2
    
    phi = np.arctan2(R[2][1], R[2][2])
    theta = -atan(R[2][1]/np.sqrt(1-R[2][0]**2))
    psi = np.arctan2(R[1][0], R[0][0])

    euler = [phi, theta, psi]
    
    return euler

def quatern2rotMat(q):
    '''
    converts a quaternion orientation to a rotation matrix
    '''
    R = [[0,0,0],[0,0,0],[0,0,0]]
    
    R[0][0] = 2*q[0]**2-1+2*q[1]**2
    R[0][1] = 2*(q[1]*q[2]+q[0]*q[3])
    R[0][2] = 2*(q[1]*q[3]-q[0]*q[2])

    R[1][0] = 2*(q[1]*q[2]-q[0]*q[3])
    R[1][1] = 2*q[0]**2-1+2*q[2]**2
    R[1][2] = 2*(q[2]*q[3]+q[0]*q[1])
        
    R[2][0] = 2*(q[1]*q[3]+q[0]*q[2])
    R[2][1] = 2*(q[2]*q[3]-q[0]*q[1])
    R[2][2] = 2*q[0]**2-1+2*q[3]**2

    return R

def quaternConj(q):
    '''converts a quaternion to its conjugate'''
    qConj = [q[0], -q[1], -q[2], -q[3]]

    return qConj

def quaternProd(a,b):
    '''calculates the quaternion product'''
    ab = [0,0,0,0]

    ab[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
    ab[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
    ab[2] = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
    ab[3] = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]

    return ab

def rotMat2euler(R):
    '''
    converts a rotation matrix orientation to ZYX euler angles where phi is
    a rotation around X, theta around Y and psi around Z
    '''    
    phi = np.arctan2(R[2][1], R[2][2])
    theta = -atan(R[2][1]/np.sqrt(1-R[2][0]**2))
    psi = np.arctan2(R[1][0], R[0][0])

    euler = [phi, theta, psi]

    return euler

def rotMat2quatern(R):
    '''
    converts a rotation matrix orientation to a quaternion
    '''
    row = 3
    col = 3
    numR = len(R)
    q = np.zeros([numR,4])
    K = np.zeros([4,4])
    for i in range(numR):
        K[0][0] = (1/3) * (R[0][0] - R[1][1] - R[2][2])
        K[0][1] = (1/3) * (R[1][0] + R[0][1])
        K[0][2] = (1/3) * (R[2][0] + R[0][2])
        K[0][3] = (1/3) * (R[1][2] - R[2][1])
        K[1][0] = (1/3) * (R[1][0] + R[0][1])
        K[1][1] = (1/3) * (R[1][1] - R[0][0] - R[2][2])
        K[1][2] = (1/3) * (R[2][1] + R[1][2])
        K[1][3] = (1/3) * (R[2][0] - R[0][2])
        K[2][0] = (1/3) * (R[2][0] + R[0][2])
        K[2][1] = (1/3) * (R[2][1] + R[1][2])
        K[2][2] = (1/3) * (R[2][2] - R[0][0] - R[1][1])
        K[2][3] = (1/3) * (R[0][1] - R[1][0])
        K[3][0] = (1/3) * (R[1][2] - R[2][1])
        K[3][1] = (1/3) * (R[2][0] - R[0][2])
        K[3][2] = (1/3) * (R[0][1] - R[1][0])
        K[3][3] = (1/3) * (R[0][0] + R[1][1] + R[2][2])
        
        [v,d] = np.linalg.eig(K)[1]
        q[i] = v[3]
        q[i] = [q[i][3], q[i][0], q[i][1], q[i][2]]

    return q

def testScript():
    '''script tests the quaternion library functions to ensure that each
    each function output is consistent'''

    # axis-angle to rotation matrix
    axis = [1,2,3]
    axis = axis/np.linalg.norm(axis)
    angle = np.pi/2
    
    R = axisAngle2rotMat(axis,angle)
    print 'axis angle to rotation matrix'
    print R

    # axis-angle to quaternion
    q = axisAngle2quatern(axis,angle)
    print 'axis-angle to quaternion'
    print q

    # quaternion to rotation matrix
    R = quatern2rotMat(q)
    print 'quaternion to rotation matrix'
    print R

    # rotation matrix to quaternion
    q = rotMat2quatern(R)
    print 'rotation matrix to quaternion'
    print q
                                   
    # rotation matrix to ZYX euler angles
    euler = rotMat2euler(R)
    print 'rotation matrix to ZYX euler angles'
    print euler
                                   
    # quaternion to ZYX euler angles
    euler = quatern2euler(q)
    print 'quaternion to ZYX euler angles'
    print euler
    
    # ZYX euler angles to rotation matrix
    R = euler2rotMat(euler[0], euler[1], euler[2])
    print 'ZYX euler angles to rotation matrix'
    print R

    return None
    
    
if __name__ == "__main__":
    testScript()
