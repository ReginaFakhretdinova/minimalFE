# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

from src.mesh.usmesh import usmesh
from src.mesh.fusmesh import fusmesh
from src.fe.TRI3_Element import *
from src.fe.SEG4_Element import *
from fe.assemble import assemble
from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix
from src.mesh.mesh_utils import project_segment_elt

import meshio
m = meshio.read("src/fe/mesh.obj")
mmesh=usmesh.fromMeshio(m, 1)
del m

# FRACTURE
xf1 = -5.0; yf1 = 0.0;
xf2 = +5.0; yf2 = +0.0;
# fracture connectivity matrix. 
j=1; index = dict({"tip1": np.array([], dtype=int), "tip2": np.array([], dtype=int), "boundary": np.array([], dtype=int)})
for i in range(0,mmesh.nnodes-2): 
    if sum((mmesh.coor[i] == mmesh.coor[j])*1) == 3: #remember duplicated nodes => FRAC NODES
        index["boundary"] = np.append(index["boundary"], int(i))
    j+=1
    if sum((mmesh.coor[i] == [xf1, yf1, 0])*1) == 3: #find tip 1
        index["tip1"] = np.append(index["tip1"], int(i))
    if sum((mmesh.coor[i] == [xf2, yf2, 0])*1) == 3: #find tip 2
        index["tip2"] = np.append(index["tip2"], int(i))

num = np.size(index["boundary"])
conn = list()
for i in range(0, num+1):
    conn.append(list())

a = np.array([]); b = np.array([]); 
for i in index["boundary"]:
    a = np.append(a, np.sum( ( mmesh.coor[index["tip1"]] -mmesh.coor[i]) **2, 1 ) )
    b = np.append(b, np.sum( ( mmesh.coor[index["tip2"]] -mmesh.coor[i]) **2, 1 ) )
a = np.argmin(a); b = np.argmin(b) # a – closest to tip1; b – closest to tip2
conn[0].append(index["tip1"][0]); conn[0].append(index["boundary"][a]); conn[0].append(index["boundary"][a] + 1)
conn[1].append(conn[0][1]); conn[1].append(conn[0][2])
index["boundary"] = index["boundary"][index["boundary"] != conn[0][1]]

for i in range(2,num):
    a = np.array([])
    for j in index["boundary"]:
        a = np.append(a, np.sum( ( mmesh.coor[conn[i-1][0]] -mmesh.coor[j]) **2) )
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
fmesh=fusmesh(mmesh.dimension, np.array([mmesh.coor[index["boundary"]]], dtype=np.int), conn, np.array([3], dtype=np.int))

## plotting the unstructured mesh
triang=matplotlib.tri.Triangulation(mmesh.coor[:,0], mmesh.coor[:,1], triangles=mmesh.conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'b-', lw=1)
ax1.plot(0,0.,'ko')
plt.show()

properties={"Conductivity": 1.,"Storage":1. }

matid=np.zeros(mmesh.nelts,dtype=int)

# assemble conductivity matrix.
t = time.process_time()
myC=assemble(mmesh,conductivityMatrix,matid,np.array([properties["Conductivity"]]), project_segment_elt)
fC=assemble(mmesh,conductivityfracMatrix,matid,np.array([properties["Conductivity"]]), project_segment_elt)
elapsed_time = time.process_time() - t
print(elapsed_time)

# assemble  mass matrix.
t = time.process_time()
myM=assemble(mmesh,massMatrix,matid,np.array([properties["Storage"]]), project_segment_elt)
elapsed_time = time.process_time() - t
print(elapsed_time)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.spy(myM[1:10,1:10])
plt.show()

# assemble  source matrix.

myF=csr_matrix((mmesh.nnodes,1))
 # just enforce 1 flux as nodal force for node located at (0.,0.)
ij = np.where(mmesh.coor[:,0]==0.) # anyway there is a
myF[ij[0][0]]=1.

# checking geometry accuracy
sum=0.
for e in range(mmesh.nelts):
    xae=mmesh.coor[mmesh.conn[e],:2]
    sum=sum+jacobian(xae)/2

abs(1-sum/(np.pi*10*10))

dt = 0.01
myK=myM + dt * myC
luK=splu(myK.tocsc())  # lu decomposition

p0=np.zeros(mmesh.nnodes,dtype=float)
pn=p0.copy()
tt=0.
k=0
q0=np.asarray([0.])
times=np.asarray([0.])

while k < 2000:
    k=k+1
    tt=tt+dt
    times=np.append(times, [tt])
    ft =dt*np.asarray(-(myC@pn)+(myF.transpose()))[0]
    dp=luK.solve(ft)
    pn=pn+dp
    if (k % 300) == 0 :
        fig1, ax1 = plt.subplots()
        ax1.tricontourf(triang, pn)
        ax1.axis('equal')
        plt.show()

rr=(mmesh.coor[:,0]**2+mmesh.coor[:,1]**2)**(0.5)

# plots and checks
#sol=pressolution(coor,tt)
import scipy.special as sc
def pres(r,t):
    return (-1./(4*np.pi))*sc.expi(-r*r/(4*t))

rs=np.linspace(0.0, 10.0, num=300)
sol=pres(rs,tt)

fig, ax = plt.subplots()
ax.plot(rs, sol)
ax.plot(rr, pn,'.')
plt.show()


# fig, ax = plt.subplots()
# ax.plot(rr, abs(pn-pres(rr,t)),'.')
# plt.show()

np.median(abs(pn-pres(rr,t)))

np.min(abs(pn-pres(rr,t)))

np.max(abs(pn-pres(rr,t)))



