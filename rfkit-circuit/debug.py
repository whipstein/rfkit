import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix

# km = 0.4
# lp = 18.7193e-12
# ls = 11.3005e-12
# zp = np.array([9.4595-28.0873j, 16.923-5.84j])
# w = 2.*np.pi*275e9
# yp = 1/zp
# rp = 1/np.real(yp)
# cp = np.imag(yp)/w
# lpx = (1-km)*lp
# lsx = (1-km)*ls
# m = km*lp
# n2 = lp/ls; n = np.sqrt(n2)

# z = np.array([ \
#     [rp[0], 1/(1j*w*cp[0]), 1j*w*lpx], \
#     [rp[1], 1/(1j*w*cp[1]), 1j*w*lsx], \
#     [1j*w*lpx, 1j*w*m, n2], \
#     [1j*w*lsx, n2, 1.] \
#     ])
# y = 1/z
# y[3,2] = 0.
# z[3,2] = 0.
# ysum = np.sum(y, axis=1)

# indices_c = np.array([\
#     [1,1],[2,2],[3,3], \
#     [5,5],[6,6], \
#     [8,8],[9,9], \
#     [10,10],[11,11], \
#     [13,13], \
#     [1,5],[5,1], \
#     [2,8],[8,2], \
#     [3,11],[11,3], \
#     [6,10],[10,6], \
#     [9,13],[13,9], \
#     [12,12], \
#     [14,14], \
#     [12,14],[14,12] \
#     ])
# data_c = np.concatenate([ \
#     np.repeat(1./3., 10), \
#     np.repeat(2./3., 10), \
#     np.array([(1-n2)/(1+n2)]), \
#     np.array([(n2-1)/(1+n2)]), \
#     np.repeat(2*n/(1+n2), 2) \
#     ])
# C = coo_matrix((data_c, (indices_c[:,0], indices_c[:,1])), shape=(15,15), dtype=complex).tocsr()

# data_x = np.zeros((4,3,3), dtype=complex)
# for i in range(4):
#     for j in range(3):
#         for k in range(3):
#             data_x[i,j,k] = 2./(z[i,k]*ysum[i])
#             if j == k:
#                 data_x[i,j,k] -= 1.

# indices_x = np.array([ \
#     [0,0],[1,1],[2,2],[3,3], \
#     [4,4],[4,5],[4,6],[5,4],[5,5],[5,6],[6,4],[6,5],[6,6], \
#     [7,7],[7,8],[7,9],[8,7],[8,8],[8,9],[9,7],[9,8],[9,9], \
#     [10,10],[10,11],[10,12],[11,10],[11,11],[11,12],[12,10],[12,11],[12,12], \
#     [13,13],[13,14],[14,13],[14,14] \
#     ])
# data_x = data_x.ravel()[:-3]
# data_x = np.delete(data_x, [-4,-1])
# data_x = np.insert(data_x, 0, [-1,-1,-1,-1])
# X = coo_matrix((data_x, (indices_x[:,0],indices_x[:,1])), shape=(15,15), dtype=complex).tocsr()
# # print(X)

nnodes = 2
nelems = 1
size = nnodes + 2 * nelems + 1
zp = np.array([50., 50.], dtype=complex)
w = 2. * np.pi * 1e9
yp = 1/zp
rp = 1/np.real(yp)
cp = np.imag(yp)/w
km = 0.4
lp = 18.7517
ls = 11.5291
n2 = lp / ls
n = np.sqrt(n2)
r = 20.
z = np.array([ \
    [rp[0], r], \
    [rp[1], r], \
    ])
y = 1/z
ysum = np.sum(y, axis=1)
# print(y)
print("Ysum:")
for v in ysum:
    print(f"vec![c64({v.real}, {v.imag})],")

# 0: gnd
# 1: p1
# 2: R
# 3: p2
# 4: R
port = [1,3]
indices_c = np.array([\
    [1,1],[3,3], \
    [1,3],[3,1], \
    ])
data_c = np.concatenate([ \
    np.repeat(1./3., 2), \
    np.repeat(2./3., 2), \
    ])
C = coo_matrix((data_c, (indices_c[:,0], indices_c[:,1])), shape=(size,size), dtype=complex)
print("C:")
for r, c, v in zip(C.row, C.col, C.data):
    print(f"({r}, {c}) -> ({v.real}, {v.imag})")

sweep = len(z[0,:])
data_x = np.zeros((nnodes,sweep,sweep), dtype=complex)
for i in range(nnodes):
    for j in range(sweep):
        for k in range(sweep):
            data_x[i,j,k] = 2./(z[i,k]*ysum[i])
            if j == k:
                data_x[i,j,k] -= 1.
            if y[i,j] == 0. or y[i,k] == 0.:
                data_x[i,j,k] = np.inf

indices_x = np.array([ \
    [1,1],[1,2],[2,1],[2,2], \
    [3,3],[3,4],[4,3],[4,4], \
    ])
data_x = data_x.ravel()
data_x = np.delete(data_x, np.argwhere(data_x == np.inf))
# data_x = np.insert(data_x, 0, [-1,-1])
X = coo_matrix((data_x, (indices_x[:,0],indices_x[:,1])), shape=(size,size), dtype=complex)
print("X:")
for r, c, v in zip(X.row, X.col, X.data):
    # print(f"({r}, {c}) -> ({v.real}, {v.imag})")
    print(f"c64({v.real}, {v.imag}),")


net = np.eye(size) - C @ X
# # indices = np.argwhere(net != 0.)
# # for r, c in indices:
# #     print(f"({r}, {c}) -> {net[r,c]}")

net_inv = np.linalg.inv(net)
# # indices = np.argwhere(net_inv != 0.)
# # for r, c in indices:
# #     print(f"({r}, {c}) -> {net_inv[r,c]}")

S = X @ net_inv
indices = np.array([[port[0],port[0]],[port[0],port[1]],[port[1],port[0]],[port[1],port[1]]])
print("S:")
for r, c in indices:
    v = S[r,c]
    print(f"c64({v.real}, {v.imag}),")
