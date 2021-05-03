import os
import itertools

from datetime import datetime


def prepare(workdir, experiment_path, parameters):

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
    configs = list(itertools.product(*params))

    config_id = 0
    experiment_instances = []

    # Create Lipizzaner configuration files for each candidate.
    for config in configs:
        
        config_id += 1
        
        # Create instances
        for i in range(1, parameters["n_executions"]+1):

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            # Create instance dir
            instance_dir = experiment_path + "/{}-C{}-E{}".format(timestamp, config_id, i)
            os.mkdir(instance_dir)

            # Create Lipizzaner general config file
            general_config_template = open(workdir + "/templates/general.yml", "rt")
            general_config = open(instance_dir + "/general.yml", "wt")

            # Find required ports: start at 5000
            if parameters["grid_size"] == 1:
                ports = "5000"
            else:
                max_port = 4999 + parameters["grid_size"]
                ports = "5000-" + str(max_port)

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

            # Append instance
            instance = {
                "config_id": config_id,
                "exec_id": i,
                "dir": instance_dir,
                "timestamp": timestamp
            }

            experiment_instances.append(instance)

    return experiment_instances
