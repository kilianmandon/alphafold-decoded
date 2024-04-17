# Rendering code from the Blog Post
#   "Custom 3D engine in Matplotlib"
# by Nicolas P. Rougier
# https://matplotlib.org/matplotblog/posts/custom-3d-engine/

import time
import torch
import math
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection

def frustum(left, right, bottom, top, znear, zfar):
    M = torch.zeros((4, 4))
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M.float()

def perspective(fovy, aspect, znear, zfar):
    h = math.tan(0.5*torch.pi / 180 * fovy) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return torch.tensor([[1, 0, 0, x], [0, 1, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=torch.float32)
def xrotate(theta):
    t = torch.pi * theta / 180
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[1, 0,  0, 0], [0, c, -s, 0],
                     [0, s,  c, 0], [0, 0,  0, 1]], dtype=torch.float32)
def yrotate(theta):
    t = math.pi * theta / 180
    c, s = math.cos(t), math.sin(t)
    return  torch.tensor([[ c, 0, s, 0], [ 0, 1, 0, 0],
                      [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=torch.float32)

class BunnyRenderer:
    def __init__(self):
        self.collection = None

    def reset(self):
        self.collection = None
        

    def render_bunny(self, V, ax=None):
        MVP = perspective(25,1,1,100) @ translate(0,0,-3.5) # @ xrotate(20) # @ yrotate(45)
        V = torch.cat((V, torch.ones((V.shape[0], 1))), dim=1)
        V = V  @ MVP.T
        V /= V[:,3].reshape(-1,1)
        V = V[self.F]
        T =  V[:,:,:2]
        Z = -V[:,:,2].mean(axis=1)
        zmin, zmax = Z.amin(), Z.amax()
        Z = (Z-zmin)/(zmax-zmin)
        C = plt.get_cmap("magma")(Z)
        I = torch.argsort(Z)
        T, C = T[I,:], C[I,:]
        if ax is None:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
        if self.collection is None:
            self.collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
            ax.add_collection(self.collection)

        self.collection.set_verts(T)
        # ax.clear()
        

    def load_bunny(self, path):
        V, F = [], []
        with open(path) as f:
            for line in f.readlines():
                if line.startswith('#'):  
                    continue
                values = line.split()
                if not values:            
                    continue
                if values[0] == 'v':      
                    V.append([float(x) for x in values[1:4]])
                elif values[0] == 'f' :   
                    F.append([int(x) for x in values[1:4]])

        V, F = torch.tensor(V), torch.tensor(F)-1
        V = (V-(V.amax(0)+V.amin(0))/2) / max(V.amax(0)-V.amin(0))

        self.F = F
        return V

   
if __name__=='__main__':
    renderer = BunnyRenderer() 
    V = renderer.load_bunny('solutions/geometry/control_values/bunny.obj')
    # phi = math.pi/4
    phi = 0
    ex = torch.tensor([math.cos(phi),0,math.sin(phi)]).float()
    ey = torch.tensor([0, 1, 0]).float()
    ex = ex/torch.linalg.vector_norm(ex)
    ey = ey - torch.dot(ex, ey) * ex
    ey = ey / torch.linalg.vector_norm(ey)
    ez = torch.linalg.cross(ex, ey)

    rot_mat = torch.stack((ex, ey, ez), dim=1)
    V = V@rot_mat
    renderer.render_bunny(V)