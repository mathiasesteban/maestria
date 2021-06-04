#!/usr/bin/env python

import os
import subprocess
import re
import sys
import pathlib
import time
import json
import itertools
import random
import filelock
import pickle

from datetime import datetime

from control import ExecutionControl


def prepare(workdir, experiment_path, ports, parameters):

    # Read parameters candidates.
    params = []
    params.append(parameters["trainer_name"])
    params.append(parameters["trainer_n_iterations"])
    params.append(parameters["trainer_calculate_net_weights_dist"])
    params.append(parameters["trainer_mixture_generator_samples_mode"])
    params.append(parameters["trainer_params_population_size"])
    params.append(parameters["trainer_params_tournament_size"])
    params.append(parameters["trainer_params_n_replacements"])
    params.append(parameters["trainer_params_default_adam_learning_rate"])
    params.append(parameters["trainer_params_alpha"])
    params.append(parameters["trainer_params_mutation_probability"])
    params.append(parameters["trainer_params_discriminator_skip_each_nth_step"])
    params.append(parameters["trainer_params_mixture_sigma"])
    params.append(parameters["trainer_params_enable_selection"])
    params.append(parameters["trainer_params_evaluate_subpopulations_every"])
    params.append(parameters["trainer_params_subpopulation_sample_size"])
    params.append(parameters["trainer_params_score_enabled"])
    params.append(parameters["trainer_params_score_type"])
    params.append(parameters["trainer_params_score_score_sample_size"])
    params.append(parameters["trainer_params_score_cuda"])
    params.append(parameters["trainer_params_fitness_fitness_sample_size"])
    params.append(parameters["trainer_params_fitness_fitness_mode"])
    params.append(parameters["trainer_params_fitness_fitness_batch_size"])
    params.append(parameters["dataloader_dataset_name"])
    params.append(parameters["dataloader_use_batch"])
    params.append(parameters["dataloader_batch_size"])
    params.append(parameters["dataloader_n_batches"])
    params.append(parameters["dataloader_shuffle"])
    params.append(parameters["dataloader_smote_augmentation_times"])
    params.append(parameters["dataloader_gaussian_augmentation_times"])
    params.append(parameters["dataloader_gaussian_augmentation_mean"])
    params.append(parameters["dataloader_gaussian_augmentation_std"])
    params.append(parameters["network_name"])
    params.append(parameters["network_loss"])
    params.append(parameters["master_calculate_score"])
    params.append(parameters["master_score_sample_size"])
    params.append(parameters["master_cuda"])

    # Generate possible tuples.
    # Code from grid search project, just one config will be generated as every list has only one element for an IRACE instance.)
    configs = list(itertools.product(*params))
    config = configs[0]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    
    # Create instance dir
    instance_dir = experiment_path + "/{}".format(timestamp)
    os.mkdir(instance_dir)

    # Create Lipizzaner general config file
    general_config_template = open(workdir + "/templates/general.yml", "rt")
    general_config = open(instance_dir + "/general.yml", "wt")

    # Find required ports: start at 5000
    #if parameters["grid_size"] == 1:
     #   ports = "5000"
    #else:
     #   max_port = 4999 + parameters["grid_size"]
      #  ports = "5000-" + str(max_port)

    #port = random.randint(1, 50)
    #ports = str(5000 + port)

    for line in general_config_template:
        newline = line.replace('OUTPUT_DIR', instance_dir)
        newline = newline.replace('PORTS', ports)
        general_config.write(newline)

    general_config_template.close()
    general_config.close()

    # Create Lipizzaner main config file
    config_template = open(str(workdir) + "/templates/main.yml", "rt")
    specific_config_path = instance_dir + "/main.yml"
    specific_config = open(specific_config_path, "wt")

    for line in config_template:
        newline = line.replace('TRAINER_NAME', config[0])
        newline = newline.replace('TRAINER_N_ITERATIONS', config[1])
        newline = newline.replace('TRAINER_CALCULATE_NET_WEIGHTS_DIST', config[2])
        newline = newline.replace('TRAINER_MIXTURE_GENERATOR_SAMPLES_MODE', config[3])
        newline = newline.replace('TRAINER_PARAMS_POPULATION_SIZE', config[4])
        newline = newline.replace('TRAINER_PARAMS_TOURNAMENT_SIZE', config[5])
        newline = newline.replace('TRAINER_PARAMS_N_REPLACEMENTS', config[6])
        newline = newline.replace('TRAINER_PARAMS_DEFAULT_ADAM_LEARNING_RATE', config[7])
        newline = newline.replace('TRAINER_PARAMS_ALPHA', config[8])
        newline = newline.replace('TRAINER_PARAMS_MUTATION_PROBABILITY', config[9])
        newline = newline.replace('TRAINER_PARAMS_DISCRIMINATOR_SKIP_EACH_NTH_STEP', config[10])
        newline = newline.replace('TRAINER_PARAMS_MIXTURE_SIGMA', config[11])
        newline = newline.replace('TRAINER_PARAMS_ENABLE_SELECTION', config[12])
        newline = newline.replace('TRAINER_PARAMS_EVALUATE_SUBPOPULATIONS_EVERY', config[13])
        newline = newline.replace('TRAINER_PARAMS_SUBPOPULATION_SAMPLE_SIZE', config[14])
        newline = newline.replace('TRAINER_PARAMS_SCORE_ENABLED', config[15])
        newline = newline.replace('TRAINER_PARAMS_SCORE_TYPE', config[16])
        newline = newline.replace('TRAINER_PARAMS_SCORE_SCORE_SAMPLE_SIZE', config[17])
        newline = newline.replace('TRAINER_PARAMS_SCORE_CUDA', config[18])
        newline = newline.replace('TRAINER_PARAMS_FITNESS_FITNESS_SAMPLE_SIZE', config[19])
        newline = newline.replace('TRAINER_PARAMS_FITNESS_FITNESS_MODE', config[20])
        newline = newline.replace('TRAINER_PARAMS_FITNESS_FITNESS_BATCH_SIZE', config[21])
        newline = newline.replace('DATALOADER_DATASET_NAME', config[22])
        newline = newline.replace('DATALOADER_USE_BATCH', config[23])
        newline = newline.replace('DATALOADER_BATCH_SIZE', config[24])
        newline = newline.replace('DATALOADER_N_BATCHES', config[25])
        newline = newline.replace('DATALOADER_SHUFFLE', config[26])
        newline = newline.replace('DATALOADER_SMOTE_AUGMENTATION_TIMES', config[27])
        newline = newline.replace('DATALOADER_GAUSSIAN_AUGMENTATION_TIMES', config[28])
        newline = newline.replace('DATALOADER_GAUSSIAN_AUGMENTATION_MEAN', config[29])
        newline = newline.replace('DATALOADER_GAUSSIAN_AUGMENTATION_STD', config[30])
        newline = newline.replace('NETWORK_NAME', config[31])
        newline = newline.replace('NETWORK_LOSS', config[32])
        newline = newline.replace('MASTER_CALCULATE_SCORE', config[33])
        newline = newline.replace('MASTER_SCORE_SAMPLE_SIZE', config[34])
        newline = newline.replace('MASTER_CUDA', config[35])

        specific_config.write(newline)

    config_template.close()
    specific_config.close()

    return instance_dir



def train_lipizzaner(grid_size, instance_path, lipizzaner_path):

    # Launch Lipizzaner clients: Popen continues execution
    clients_pool = []
    for i in range(grid_size):
        client_command = ["python", lipizzaner_path, "train", "--distributed", "--client", "-f", instance_path + "/main.yml"]
        lipizzaner_client = subprocess.Popen(client_command,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clients_pool.append(lipizzaner_client)
        # Wait to initialize next clients and master process.
        time.sleep(30)

    # Launch Lipizzaner master: Run waits for the command to finish
    master_command = ["python", lipizzaner_path, "train", "--distributed", "--master", "-f", instance_path + "/main.yml"]
    lipizzaner_master = subprocess.run(master_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Kill clients
    for client in clients_pool:
        client.kill()

    # Parse score
    match = re.search('Best result:.* = \((.*), (.*)\)', lipizzaner_master.stderr)
    score = None

    if match is not None:
        score = float(match.group(1))

    return score


if __name__ == "__main__":

    workdir = str(pathlib.Path(__file__).parent.absolute())

    lipizzaner_path = str(pathlib.Path(__file__).parent.absolute()) + "/../lipizzaner/src/main.py"

    # Create output folder
    experiment_path = workdir + "/../results/irace"

    # Get Lipizzaner parameters file
    parameters_file = open(str(workdir) + "/templates/parameters.json", 'r')
    parameters = json.loads(parameters_file.read())

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # Parse parameters
    while cand_params:
        
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        value = cand_params.pop(0)

        # Receive values from IRACE and loaded in parameters dic.
        if param == "--trainer_params_alpha":
            parameters["trainer_params_alpha"] = [value]

        elif param == "--trainer_params_default_adam_learning_rate":
            parameters["trainer_params_default_adam_learning_rate"] = [value]
        
        elif param == "--trainer_params_mutation_probability":
            parameters["trainer_params_mutation_probability"] = [value]
        
        elif param == "--trainer_params_discriminator_skip_each_nth_step":
            parameters["trainer_params_discriminator_skip_each_nth_step"] = [value]

        else:
            print("Unknown parameter %s" % param)
            sys.exit(1)

    execution_control_file_path = experiment_path + '/control.pkl'
    my_pid = os.getpid()

    # Check if control file exists
    if not os.path.isfile(execution_control_file_path):

        # Lock process to create control file
        lock = filelock.FileLock("{}.lock".format(execution_control_file_path))
        with lock.acquire():

            # Check again after locking to avoid creating twice
            if not os.path.isfile(execution_control_file_path):

                execution_control = ExecutionControl(my_pid, parameters["min_port"], parameters["max_port"],
                                                     parameters["grid_size"], parameters["required_mem"])

                with open(execution_control_file_path, "wb+") as control_pkl:
                    pickle.dump(execution_control, control_pkl)

            lock.release()

    # Add myself to queue
    lock = filelock.FileLock("{}.lock".format(execution_control_file_path))
    with lock.acquire():

        with open(execution_control_file_path, "rb+") as execution_control_file:
            execution_control = pickle.load(execution_control_file)
            execution_control.add_to_queue(my_pid)

        with open(execution_control_file_path, "rb+") as execution_control_file:
            pickle.dump(execution_control, execution_control_file)

        lock.release()

    # Wait to launch
    ready_to_launch = False
    while not ready_to_launch:

        random_sleep = random.randint(15, 30)
        time.sleep(random_sleep)

        lock = filelock.FileLock("{}.lock".format(execution_control_file_path))
        with lock.acquire():

            # Load controller
            execution_control_file_read_stream = open(execution_control_file_path, "rb+")
            execution_control = pickle.load(execution_control_file_read_stream)
            execution_control_file_read_stream.close()

            # Ask for permition
            execution_resources = execution_control.is_my_turn(my_pid)

            # Save controller
            execution_control_file_write_stream = open(execution_control_file_path, "rb+")
            pickle.dump(execution_control, execution_control_file_write_stream)
            execution_control_file_write_stream.close()

            if execution_resources is not None:
                ready_to_launch = True

            lock.release()

    # Prepare
    instance_path = prepare(workdir, experiment_path, execution_resources["ports"],  parameters)

    # Execute
    start_time = time.time()
    score = train_lipizzaner(parameters["grid_size"], instance_path, lipizzaner_path)
    end_time = time.time()
    wall_clock = end_time - start_time

    # Target runner must PRINT cost function.
    # IRACE minimizes cost function, if we have a positive score we should return the oposite to maximize it.
    # If the score is None we penalize this instance.
    if parameters["is_maximization"]:
        if score is None:
            score = 999999
        else:
            score = -1 * score
    else:
        if score is None:
            score = -999999

    # Free resources
    lock = filelock.FileLock("{}.lock".format(execution_control_file_path))

    with lock.acquire():

        # Load controller
        execution_control_file_read_stream = open(execution_control_file_path, "rb+")
        execution_control = pickle.load(execution_control_file_read_stream)
        execution_control_file_read_stream.close()

        # Free resources
        execution_control.free_resources(my_pid, score, wall_clock)

        # Save controller
        execution_control_file_write_stream = open(execution_control_file_path, "rb+")
        pickle.dump(execution_control, execution_control_file_write_stream)
        execution_control_file_write_stream.close()

        lock.release()

    print(score)
