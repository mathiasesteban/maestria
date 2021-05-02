import subprocess
import re
import time


def run(config, experiment_instances):

    experiment_results = []

    # Execute lipizzaner for each experiment
    for instance in experiment_instances:

        start = time.time()

        # Launch clients
        clients_pool = []

        for j in range(config["grid_size"]):
            client_command = ["python", config["lipizzaner_path"], "train", "--distributed", "--client"]

            # Important to send output to DEVNULL in order to not populate SLURM file in ClusterUY.
            # Output info is logged by Lipizzaner.
            lipizzaner_client = subprocess.Popen(client_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clients_pool.append(lipizzaner_client)
            # Important to wait for clients before initializing master
            time.sleep(60)

        # Launch master
        master_command = ["python",
                          config["lipizzaner_path"],
                          "train",
                          "--distributed",
                          "--master",
                          "-f", instance["dir"] + "/lipizzaner-main-config.yml"]

        lipizzaner_master = subprocess.run(master_command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)

        end = time.time()
        wall_clock = end - start

        # Save master output for easy access
        master_stderr = open(instance["dir"] + "/master_stderr.log", "wt")
        master_stderr.write(str(lipizzaner_master.stderr))
        master_stderr.close()

        # Kill clients
        for client in clients_pool:
            client.kill()

        # Search FID
        if lipizzaner_master.returncode != 0:
            fid = "Non-cero exit code ({})".format(lipizzaner_master.returncode)
        else:
            match = re.search('Best result:.* = \((.*), (.*)\)', lipizzaner_master.stderr)
            if match is None:
                fid = "not found"
            else:
                fid = match.group(1)

        instance_result = {
            "timestamp":  instance["timestamp"],
            "config": instance["config"],
            "exec": instance["exec"],
            "wall_clock": wall_clock,
            "fid": fid
        }

        result_line = "{}-C{}E{}: {} ({})".format(instance_result["timestamp"],
                                                                 instance_result["config"],
                                                                 instance_result["exec"],
                                                                 instance_result["fid"],
                                                                 instance_result["wall_clock"])

        # In order to see instance results in SLURM file before the experiment finishes
        print(result_line)

        experiment_results.append(instance_result)

    return experiment_results
