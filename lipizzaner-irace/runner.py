#!/usr/bin/env python

import os
import subprocess
import re
import sys
import pathlib
import time

from datetime import datetime

# Fixed params
LIPIZZANER_PATH = "/home/mesteban/git/lipizzaner-covidgan/src/main.py"
GRID_SIZE = 1
DIR = pathlib.Path(__file__).parent.absolute()


def train_lipizzaner(batch_size, network, smote_size, mutation_probabilitie, adam_rate):

    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = str(DIR) + "/outputs/" + timestamp
    os.mkdir(output_path)

    # Create general configuration file
    general_config_template = open(str(DIR) + "/lipizzaner-general-config-template.yml", "rt")
    general_config = open(output_path + "/lipizzaner-general-config.yml", "wt")

    if GRID_SIZE == 1:
        ports = "5000"
    else:
        max_port = 4999 + GRID_SIZE
        ports = "5000-" + str(max_port)

    for line in general_config_template:
        newline = line.replace('OUTPUT_DIR', output_path)
        newline = newline.replace('PORTS', ports)
        general_config.write(newline)

    general_config_template.close()
    general_config.close()

    # Create specific config file
    main_config_template = open(str(DIR) + "/lipizzaner-main-config-template.yml", "rt")
    main_config_path = output_path + "/lipizzaner-main-config-template.yml.yml"
    config = open(main_config_path, "wt")

    for line in main_config_template:

        newline = line.replace('DEFAULT_ADAM_LEARNING_RATE', str(adam_rate))
        newline = newline.replace('MUTATION_PROBABILITY', str(mutation_probabilitie))
        newline = newline.replace('BATCH_SIZE', str(batch_size))
        newline = newline.replace('SMOTE_AUGMENTATION_TIMES', str(smote_size))
        newline = newline.replace('NETWORK_NAME', network)

        config.write(newline)

    main_config_template.close()
    config.close()

    # Launch Lipizzaner clients: Popen continues execution
    clients_pool = []
    for i in range(GRID_SIZE):
        client_command = ["python", LIPIZZANER_PATH, "train", "--distributed", "--client"]
        lipizzaner_client = subprocess.Popen(client_command,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clients_pool.append(lipizzaner_client)
        # Wait to initialize next clients and master process.
        time.sleep(20)

    # Launch Lipizzaner master: Run waits for the command to finish
    master_command = ["python", LIPIZZANER_PATH, "train", "--distributed", "--master", "-f", main_config_path]
    lipizzaner_master = subprocess.run(master_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Kill clients
    for client in clients_pool:
        client.kill()

    # Parse FID score
    match = re.search('Best result:.* = \((.*), (.*)\)', lipizzaner_master.stderr)
    if match is not None:
        fid = float(match.group(1))
    else:
        fid = -1

    return fid


if __name__ == "__main__":

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # Default values (if any)
    batch_size = None
    network = None
    smote_size = None
    mutation_probabilitie = None
    adam_rate = None

    # Parse parameters
    while cand_params:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        if param == "--batch_size":
            batch_size = int(value)
        elif param == "--network":
            network = value
        elif param == "--smote_size":
            smote_size = int(value)
        elif param == "--mutation_probabilitie":
            mutation_probabilitie = float(value)
        elif param == "--adam_rate":
            adam_rate = float(value)
        else:
            print("Unknown parameter %s" % param)
            sys.exit(1)

    # Sanity checks
    if batch_size is None or \
            network is None or \
            adam_rate is None or \
            smote_size is None or \
            mutation_probabilitie is None:
        print("At least one parameter was not defined")
        sys.exit(1)

    fid_score = train_lipizzaner(batch_size, network, smote_size, mutation_probabilitie, adam_rate)

    # Target runner must print COST function.
    print(fid_score)

