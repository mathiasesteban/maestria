import os
from datetime import datetime


def prepare(workdir, experiment_path, config):

    lipizzaner_config = config["lipizzaner_config"]["specific"]

    config_id = 0
    experiment_instances = []

    # TODO: Parametrize this with KEY-VALUE style for every lipizzaner parameter
    # Config combinations
    for network in lipizzaner_config['networks']:
        for batch_size in lipizzaner_config['batch_sizes']:
            for smote_size in lipizzaner_config['smote_sizes']:
                for mutation_probability in lipizzaner_config['mutations_probabilities']:
                    for adam_rate in lipizzaner_config['adam_rates']:

                        config_id += 1

                        # Create instance
                        for i in range(1, config["n_executions"]+1):

                            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

                            # Create instance dir
                            instance_dir = experiment_path + "/{}-C{}E{}".format(timestamp, config_id, i)
                            os.mkdir(instance_dir)

                            # Create Lipizzaner general config file
                            general_config_template = open(workdir + "/lipizzaner-general-config-template.yml", "rt")
                            general_config = open(instance_dir + "/lipizzaner-general-config.yml", "wt")

                            if config["grid_size"] == 1:
                                ports = "5000"
                            else:
                                max_port = 4999 + config["grid_size"]
                                ports = "5000-" + str(max_port)

                            for line in general_config_template:
                                newline = line.replace('OUTPUT_DIR', instance_dir)
                                newline = newline.replace('PORTS', ports)
                                general_config.write(newline)

                            general_config_template.close()
                            general_config.close()

                            # Create Lipizzaner specific config file
                            config_template = open(str(workdir) + "/lipizzaner-main-config-template.yml", "rt")
                            specific_config_path = instance_dir + "/lipizzaner-main-config.yml"
                            specific_config = open(specific_config_path, "wt")

                            for line in config_template:
                                newline = line.replace('DEFAULT_ADAM_LEARNING_RATE', str(adam_rate))
                                newline = newline.replace('MUTATION_PROBABILITY', str(mutation_probability))
                                newline = newline.replace('BATCH_SIZE', str(batch_size))
                                newline = newline.replace('SMOTE_AUGMENTATION_TIMES', str(smote_size))
                                newline = newline.replace('NETWORK_NAME', network)
                                specific_config.write(newline)

                            config_template.close()
                            specific_config.close()

                            # Append instance
                            instance = {
                                "config": config_id,
                                "exec": i,
                                "dir": instance_dir,
                                "timestamp": timestamp
                            }
                            experiment_instances.append(instance)

    return experiment_instances
