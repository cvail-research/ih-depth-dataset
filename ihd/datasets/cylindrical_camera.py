import math
import numpy as np

#read csv from ouster studio
def read_points(file):

    points = []
    reflectivity = []
    zero = np.array([0.0, 0.0, 0.0])
    import csv
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) #skip header
        for row in reader:
            point = np.array([float(row[0]), float(row[1]), float(row[2])])
            if (point != zero).all():
                points.append(point)
                reflectivity.append(float(row[4]))
                
    return points, reflectivity


def read_ply(file):
    points = []
    zero = np.array([0.0, 0.0, 0.0])
    with open(file, 'r') as ptfile:
        numpts = 0
        isheader = True
        for row in ptfile:
            split = row.strip().split(' ')
           
            if not isheader:
                point = np.array([float(split[0]), float(split[1]), float(split[2])])
                if (point != zero).all():
                    points.append(point)
                continue
            
            if split[0]=='element' and split[1]=='vertex':
                numpts = int(split[2])
                
            if split[0] == 'end_header':
                isheader = False
            
    return points


def write_ply(file, points):
    with open(file, 'w') as ply:
        ply.write('ply\n')
        ply.write('format ascii 1.0\n')
        ply.write('element vertex %d\n'%len(points))
        ply.write('property float x\n')
        ply.write('property float y\n')
        ply.write('property float z\n')
        ply.write('end_header\n')
        
        for pt in points:
            ply.write('%s %s %s\n'%(repr(pt[0]), repr(pt[1]), repr(pt[2])))
    

class camera:
    def __init__(self, R, w, y, f, j0, Rot, t):
        self.R: float = R          #radius
        self.w: float = w          #principal angler
        self.y: float = y          #pixel width angle
        self.f: float = f          #focal length
        self.j0: float = j0        #principal row
        self.Rot: np.ndarray = Rot #rotation (shape=(3,3))
        self.t: np.ndarray = t     #translation (shape=(3,))


def project(P, cam):
    
    P_0 = cam.Rot.dot(P) + cam.t

    X0 = P_0[0]
    Y0 = P_0[1]
    Z0 = P_0[2]
    
    #project P onto ZX plane of camera (X0, 0, Z0)
    i_angle = np.arctan(X0 / Z0) - cam.w + np.arcsin((cam.R * np.sin(cam.w)) / (np.sqrt(X0*X0 + Z0*Z0)))    

    if Z0 < 0.0:
        i_angle += np.pi
        
    if i_angle > 2*np.pi:
        i_angle -= 2*np.pi
        
    i = i_angle / cam.y
    
    B = np.arctan(Y0 / (np.sqrt(X0*X0 + Z0*Z0 - (cam.R*cam.R) * (np.sin(cam.w)*np.sin(cam.w))) - cam.R*np.cos(cam.w)))
    j = cam.f * np.tan(B) + cam.j0
    return i,j


def project_vect(P: np.ndarray, cam: camera):
    """
    :param P: Matrix of points expected to be of shape [nPoints, 3].
    :param cam: Camera intrinsics.
    """
    P_0 = cam.Rot.dot(P.T).T + cam.t
    X0 = P_0.T[0]
    Y0 = P_0.T[1]
    Z0 = P_0.T[2]
    i_angle = np.arctan(X0 / Z0) - cam.w + np.arcsin((cam.R * np.sin(cam.w)) / (np.sqrt(X0*X0 + Z0*Z0)))
    i_angle[np.argwhere(Z0 < 0)] += np.pi
    i_angle[np.argwhere(i_angle > (2*np.pi))] -= (2*np.pi)
    i = i_angle / cam.y
    B = np.arctan(Y0 / (np.sqrt(X0*X0 + Z0*Z0 - (cam.R*cam.R) * (np.sin(cam.w)*np.sin(cam.w))) - cam.R*np.cos(cam.w)))
    j = cam.f * np.tan(B) + cam.j0
    return np.vstack([i, j]).T


def backproject2(p2d, cam):
    i_angle = p2d[0] * cam.y
    Ri = np.array([[np.cos(i_angle), 0, np.sin(i_angle)],
                   [            0, 1,              0],
                   [-np.sin(i_angle), 0, np.cos(i_angle)]])
    Rw = np.array([[np.cos(cam.w), 0, np.sin(cam.w)],
                   [            0, 1,              0],
                   [-np.sin(cam.w), 0, np.cos(cam.w)]])
    p = cam.Rot.transpose().dot(Ri.dot(Rw.dot(np.array([0,p2d[1] - cam.j0, cam.f])) + np.array([0,0,cam.R]))-cam.t)
    return p


def backproject2_vect(points: np.ndarray, cam: camera):
    """
    :param points: Matrix of 2D points in [nPoints, 2] shape.
    :param cam: Camera intrinsics.
    """
    i_angle = points[:, 0] * cam.y
    n_pts = points.shape[0]
    n_zeros = np.zeros(n_pts)

    # Create Ri from i_angle vector,[3, 3, n_pts]
    Ri = np.asarray([[np.cos(i_angle),  n_zeros,         np.sin(i_angle)],
                     [n_zeros,          np.ones(n_pts),          n_zeros],
                     [-np.sin(i_angle), n_zeros,         np.cos(i_angle)]])
    # Rw uses constant cam.w value, so we can stack the same matrix n_pts times
    # [3, 3]
    Rw = np.asarray([[np.cos(cam.w),  0, np.sin(cam.w)],
                     [0,              1,             0],
                     [-np.sin(cam.w), 0, np.cos(cam.w)]])

    # cam.R.T * ( Ri * ( ( Rw * [0, j-j0, f] ) + [0, 0, R] ) - t )
    j = points[:, 1]
    f = np.full(n_pts, cam.f)
    j_diff = np.array([n_zeros, j - cam.j0, f])
    pts_Rw = (Rw.dot(j_diff) + np.array([[0], [0], [cam.R]]))
    # Using Einsum notation to describe performing the batch matrix dot product
    pts_Ri = np.einsum("ijb,jb->ib", Ri, pts_Rw)
    p = cam.Rot.T.dot((pts_Ri.T - cam.t).T).T
    return p


def read_cam(file):
    
    camfile = open(file, 'r')
    lines = camfile.readlines()
    rotvals = []
    for i in range(3):
        rotvals.append(np.float64(lines[i].strip().split()))    
    rotation = np.asarray(rotvals)
    t = np.float64(lines[3].strip().split())
    R = float(lines[4])
    w = float(lines[5])
    f = float(lines[6])
    j0 = float(lines[7])
    y = float(lines[8])
    return camera(R, w, y, f, j0, rotation, t)


def project_vect_safe(P: np.ndarray, cam: camera):
    """
    Robust cylindrical projection with domain clipping.
    Returns (N,2) array; invalid points get NaNs.
    """
    # World -> camera
    P0 = (cam.Rot @ P.T).T + cam.t
    X0 = P0[:,0]; Y0 = P0[:,1]; Z0 = P0[:,2]

    hyp = np.sqrt(X0*X0 + Z0*Z0)
    hyp_safe = np.clip(hyp, 1e-9, None)

    sinw = math.sin(cam.w)
    arg = (cam.R * sinw) / hyp_safe
    arg = np.clip(arg, -1.0, 1.0)

    i_angle = np.arctan2(X0, Z0) - cam.w + np.arcsin(arg)
    # Adjust for behind camera (optional if model expects)
    # i_angle[Z0 < 0] += np.pi
    i_angle = np.mod(i_angle, 2*math.pi)
    i = i_angle / cam.y

    inner = hyp*hyp - (cam.R*cam.R)*(sinw*sinw)
    inner = np.clip(inner, 1e-9, None)

    denom = np.sqrt(inner) - cam.R*math.cos(cam.w)
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    B = np.arctan2(Y0, denom)
    j = cam.f * np.tan(B) + cam.j0

    ij = np.stack([i, j], axis=1)

    # Mark invalid any points where hyp was tiny (near camera center) -> set NaNs
    invalid = (hyp < 1e-6)
    if invalid.any():
        ij[invalid] = np.nan
    return ij