#
# This file is part of minimalFE.
#
# Created by Brice Lecampion on 02.07.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

# ZERO THICKNESS 4 NODES segment - following the formulation of Segura & Carol

## this is a pure python file - needs to be numba @ jit to WORK with assemble.
import numpy as np
from numba import jit

from src.fe.SEG2_Element import *

''' # FRACTURE (from mesh.obj)
xf1 = +7.0; yf1 = +10.0;
xf2 = +13.0; yf2 = +10.0;

mesh = meshio.read("src/fe/mesh.obj")
fmesh=usmesh.fromMeshio(mesh, 1)
del mesh

# fracture connectivity matrix. 
j=1; index = dict({"tip1": np.array([], dtype=int), "tip2": np.array([], dtype=int), "boundary": np.array([], dtype=int)})
for i in range(0,fmesh.nnodes-2): 
    if sum((fmesh.coor[i] == fmesh.coor[j])*1) == 3: #remember duplicated nodes => FRAC NODES
        index["boundary"] = np.append(index["boundary"], int(i))
    j+=1
    if sum((fmesh.coor[i] == [xf1, yf1, 0])*1) == 3: #find tip 1
        index["tip1"] = np.append(index["tip1"], int(i))
    if sum((fmesh.coor[i] == [xf2, yf2, 0])*1) == 3: #find tip 2
        index["tip2"] = np.append(index["tip2"], int(i))

num = np.size(index["boundary"])
conn = list()
for i in range(0, num+1):
    conn.append(list())

a = np.array([]); b = np.array([]); 
for i in index["boundary"]:
    a = np.append(a, np.sum( ( fmesh.coor[index["tip1"]] -fmesh.coor[i]) **2, 1 ) )
    b = np.append(b, np.sum( ( fmesh.coor[index["tip2"]] -fmesh.coor[i]) **2, 1 ) )
a = np.argmin(a); b = np.argmin(b) # a – closest to tip1; b – closest to tip2
conn[0].append(index["tip1"][0]); conn[0].append(index["boundary"][a]); conn[0].append(index["boundary"][a] + 1)
conn[1].append(conn[0][1]); conn[1].append(conn[0][2])
index["boundary"] = index["boundary"][index["boundary"] != conn[0][1]]

for i in range(2,num):
    a = np.array([])
    for j in index["boundary"]:
        a = np.append(a, np.sum( ( fmesh.coor[conn[i-1][0]] -fmesh.coor[j]) **2) )
    a = np.argmin(a)
    conn[i-1].append(index["boundary"][a])
    conn[i-1].append(index["boundary"][a] + 1)
    index["boundary"] = index["boundary"][index["boundary"] != conn[i-1][2]]
    conn[i].append(conn[i-1][2]); conn[i].append(conn[i-1][3])

conn[num-1].append(index["boundary"][0])
conn[num-1].append(index["boundary"][0] + 1)
conn[num].append(index["tip2"][0])
conn[num].append(conn[num-1][2])
conn[num].append(conn[num-1][3])
print(conn)
 '''
''' # FRACTURE (from mesh.obj)
xf1 = +7.0; yf1 = +10.0;
xf2 = +13.0; yf2 = +10.0;

mesh = meshio.read("src/fe/mesh.obj")
fmesh=usmesh.fromMeshio(mesh, 1)
del mesh

# fracture connectivity matrix. 
j=1; index = dict({"tip1": np.array([], dtype=int), "tip2": np.array([], dtype=int), "boundary": np.array([], dtype=int)})
for i in range(0,fmesh.nnodes-2): 
    if sum((fmesh.coor[i] == fmesh.coor[j])*1) == 3: #remember duplicated nodes => FRAC NODES
        index["boundary"] = np.append(index["boundary"], int(i))
    j+=1
    if sum((fmesh.coor[i] == [xf1, yf1, 0])*1) == 3: #find tip 1
        index["tip1"] = np.append(index["tip1"], int(i))
    if sum((fmesh.coor[i] == [xf2, yf2, 0])*1) == 3: #find tip 2
        index["tip2"] = np.append(index["tip2"], int(i))

num = np.size(index["boundary"])
conn = list()
for i in range(0, num+1):
    conn.append(list())

a = np.array([]); b = np.array([]); 
for i in index["boundary"]:
    a = np.append(a, np.sum( ( fmesh.coor[index["tip1"]] -fmesh.coor[i]) **2, 1 ) )
    b = np.append(b, np.sum( ( fmesh.coor[index["tip2"]] -fmesh.coor[i]) **2, 1 ) )
a = np.argmin(a); b = np.argmin(b) # a – closest to tip1; b – closest to tip2
conn[0].append(index["tip1"][0]); conn[0].append(index["boundary"][a]); conn[0].append(index["boundary"][a] + 1)
conn[1].append(conn[0][1]); conn[1].append(conn[0][2])
index["boundary"] = index["boundary"][index["boundary"] != conn[0][1]]

for i in range(2,num):
    a = np.array([])
    for j in index["boundary"]:
        a = np.append(a, np.sum( ( fmesh.coor[conn[i-1][0]] -fmesh.coor[j]) **2) )
    a = np.argmin(a)
    conn[i-1].append(index["boundary"][a])
    conn[i-1].append(index["boundary"][a] + 1)
    index["boundary"] = index["boundary"][index["boundary"] != conn[i-1][2]]
    conn[i].append(conn[i-1][2]); conn[i].append(conn[i-1][3])

conn[num-1].append(index["boundary"][0])
conn[num-1].append(index["boundary"][0] + 1)
conn[num].append(index["tip2"][0])
conn[num].append(conn[num-1][2])
conn[num].append(conn[num-1][3])
print(conn)
 '''
''' @jit(nopython=True)
def conductivityfracMatrix(xae,cond):
    DNaDxi = np.array([[-1., 1., 0.], [-1., 0., 1.]])
    DxDxi = DNaDxi@xae
    jacobian = np.linalg.det(DxDxi)
    DxiDx = np.linalg.inv(DxDxi)
    GradN =  DxiDx@DNaDxi
    # order 1 element - constant gradient - single gauss point
    # assuming isotropy
    Wl=1./2.
    Celt = Wl* (np.transpose(GradN)@GradN)
    return cond * jacobian * Celt

@jit(nopython=True,signature_or_function='float64[:,:](float64[:],float64)')
def conductivityMatrix(xae,cond):
    DNaDxi = np.array([[-0.5 , 0.5]])
    j = DNaDxi@xae
    GradN =   DNaDxi /j
    # order 1 element - constant gradient - single gauss point
    # assuming isotropy
    Wl=2.
    Celt = Wl* (np.transpose(GradN)@GradN) * j
    return cond  * Celt '''


# we need to code up
# the element conductivity matrix for the 4 nodes segment
# the elememt mass/storage matrix for the 4 nodes segment
# using the element matrix of the SEG2 element

# we probably need to have a special case for the "tip" element - where we only have 3 nodes for pressure
# because 
