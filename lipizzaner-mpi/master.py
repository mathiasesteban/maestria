from mpi.tags import ENTRENAR, FIN
from mpi4py import MPI
import json
import os


def master(comm, rank, size, status, output_path, grid_size):

    # Crear directorio de salida
    os.mkdir(output_path)

    # Cargar archivo de configuracion
    config_file = open("templates/config.json", 'r')
    config_json = json.loads(config_file.read())

    configs = []
    for network in config_json['networks']:
        for batch_size in config_json['batch_sizes']:
            for smote_size in config_json['smote_sizes']:
                for mutation_probability in config_json['mutations_probabilities']:
                    for adam_rate in config_json['adam_rates']:
                        new_config = {
                            'batch_size': batch_size,
                            'network': network,
                            'smote_size': smote_size,
                            'mutation_probability': mutation_probability,
                            'adam_rate': adam_rate,
                        }
                        configs.append(new_config)

    # Crear archivo config general lipizzaner
    plantilla_general = open("templates/plantilla_general.yml", "rt")
    general = open(output_path + "/general.yml", "wt")

    if grid_size == 1:
        ports = "5000"
    else:
        max_port = 4999 + grid_size
        ports = "5000-" + str(max_port)

    for line in plantilla_general:
        newline = line.replace('OUTPUT_DIR', output_path)
        newline = newline.replace('PORTS', ports)
        general.write(newline)

    plantilla_general.close()
    general.close()

    # Crear archivo de log
    log_file = open(output_path + "/log_file.txt", 'a')
    log_file.write("Se ejecutaran " + str(len(configs)) + " configuraciones\n")
    log_file.write("*******************************************************\n")

    for idx, c in enumerate(configs):
        aux_string = "{}) network: {} - batch_size: {} - smote_size: {} - mutation_probability: {} - adam_rate: {}\n"
        config_line = aux_string.format(idx,
                                        c['network'],
                                        c['batch_size'],
                                        c['smote_size'],
                                        c['mutation_probability'],
                                        c['adam_rate'])
        log_file.write(config_line)

    log_file.write("*******************************************************\n")

    results = []
    config_index = 0

    # Enviar primer lote de trabajo
    for i in range(1, size):

        # Si la cantidad de configuraciones es menor a la cantidad de procesos.
        if config_index == len(configs):
            break

        data_to_send = {'config_index': config_index, 'config': configs[config_index]}
        comm.send(data_to_send, dest=i, tag=ENTRENAR)

        config_index += 1

    while True:

        # Si no hay mas configuraciones por ejecutar corto el loop
        if config_index == len(configs):
            break

        # Espero resultado de un worker
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        results.append(result)

        log_file.write('Configuracion {} FID: {} (exit code {})\n'.format(result['config_index'],
                                                                          result['fid'],
                                                                          result['exit_code']))

        # Le envio nuevo trabajo
        data_to_send = {'config_index': config_index, 'config': configs[config_index]}
        comm.send(data_to_send, dest=status.Get_source(), tag=ENTRENAR)

        # Aumento iterador de la coleccion
        config_index += 1

    # Espero el ultimo lote de trabajo
    for rank in range(1, size):
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        log_file.write('Configuracion {} FID: {} (exit code {})\n'.format(result['config_index'],
                                                                          result['fid'],
                                                                          result['exit_code']))

        results.append(result)

    # Envio señal de fin a todos los workers
    for rank in range(1, size):
        comm.send('', dest=rank, tag=FIN)

    log_file.close()

    return results

