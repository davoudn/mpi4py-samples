from mpi4py import MPI
import numpy as np

class myclass:
   def __init__(self, position_):
        self.position = position_


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 12

data = ['a','b','c','d']


if rank == 0:
    scatter_data = data
else:
    scatter_data = None

scatter_data = comm.scatter(scatter_data, root=0)
#assert data == (rank+1)**2
#print(f"{rank}--> {scatter_data}")


data = [myclass((3,4)), myclass((8,2)), myclass((6,8)), myclass((1,5))]

if rank == 0:
    scatter_data = data
else:
    scatter_data = None

scatter_data = comm.scatter(scatter_data, root=0)

print(f"{rank}--> {scatter_data.position}")
