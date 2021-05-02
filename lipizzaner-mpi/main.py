from mpi4py import MPI
from mpi.slave import lipizzaner_slave
from mpi.master import master
from datetime import datetime


def parallel_tunning():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # 1 + slaves_number
    status = MPI.Status()

    # Params
    lipizzaner_path = "/home/mesteban/git/lipizzaner-covidgan/src/main.py"
    grid_size = 1

    # Creo el directorio de salida para el entrenamiento distribuido
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = "./generated/" + timestamp

    if rank == 0:
        master(comm, rank, size, status, output_path, grid_size)
    else:
        lipizzaner_slave(comm, rank, size, status, output_path, lipizzaner_path, grid_size)


if __name__ == "__main__":
    parallel_tunning()
