from mpi4py import MPI
import numpy as np
import time 

def matvec(comm, A, x):
    m = A.shape[0] # local rows
    p = comm.Get_size()
    rank = comm.Get_rank();
    local_dim = int(m/p)
    y_local = np.zeros( local_dim, dtype='d')

    for i in range(local_dim):
        y_local[i] = np.dot(A[i+ local_dim*rank], x)
    
    if rank == 0:
       y = np.zeros(m, dtype='d')
    else:
       y = None

    comm.Gather(y_local, y)
    comm.Bcast(y_local, root=0)

    return y

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 9000
A = np.zeros((N,N))
x = np.zeros(N)
for i in range(N):
    A[i] = i;
    x[i] = 1;
if rank == 0 :
    tm = time.process_time()
y  = matvec(comm, A, x)

if rank == 0 :
   dt = time.process_time()-tm;
   print (f"{size}: duration {dt}")

