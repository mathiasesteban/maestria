from mpi4py import MPI
from mpi.tags import FIN
import subprocess
import re


def lipizzaner_slave(comm, rank, size, status, output_path, lipizzaner_path, grid_size):

    while True:
        # Esperar datos de ejecucion
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        # Si la etiqueta es de fin termino el loop
        if status.Get_tag() == FIN:
            break

        # Creo el archivo de configuracion
        plantilla_config = open("templates/plantilla_config.yml", "rt")
        config_file_path = output_path + "/config" + str(data['config_index']) + ".yml"
        config = open(config_file_path, "wt")

        for line in plantilla_config:

            newline = line.replace('DEFAULT_ADAM_LEARNING_RATE', str(data['config']['adam_rate']))
            newline = newline.replace('MUTATION_PROBABILITY', str(data['config']['mutation_probability']))
            newline = newline.replace('BATCH_SIZE', str(data['config']['batch_size']))
            newline = newline.replace('SMOTE_AUGMENTATION_TIMES', str(data['config']['smote_size']))
            newline = newline.replace('NETWORK_NAME', data['config']['network'])

            config.write(newline)

        plantilla_config.close()
        config.close()

        # Execute Lipizzaner
        clients_pool = []
        for i in range(grid_size):
            client_command = ["python", lipizzaner_path, "train", "--distributed", "--client"]
            lipizzaner_client = subprocess.Popen(client_command)
            clients_pool.append(lipizzaner_client)

        master_command = ["python", lipizzaner_path, "train", "--distributed", "--master", "-f", config_file_path]
        lipizzaner_master = subprocess.run(master_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        for client in clients_pool:
            client.kill()

        # Obtener FID score
        match = re.search('Best result:.* = \((.*), (.*)\)', lipizzaner_master.stderr)
        if match is not None:
            fid = match.group(1)
        else:
            fid = "None"

        resultado = {
            'config_index': data['config_index'],
            'exit_code': lipizzaner_master.returncode,
            'fid': str(fid)
        }

        # Enviar el resultado a master
        comm.send(resultado, dest=0, tag=0)
