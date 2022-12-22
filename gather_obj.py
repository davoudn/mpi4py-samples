from mpi4py import MPI
import numpy as np

class myclass:
   def __init__(self, position_):
        self.position = position_


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


data = myclass((rank,rank**2)) 

data = comm.gather(data, root=0)


if rank==0 :
    for i in range(size):
        print(f"{i}--> {data[i].position}")
