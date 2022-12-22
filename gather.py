from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 10

data = np.zeros(N)

for i in range(N):
    data[i] = rank*N+i

gather_data = comm.gather(data, root=0)

print(f" {rank} {gather_data}")
