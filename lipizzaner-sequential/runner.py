import subprocess
import re
import time
import pathlib

def run(grid_size, experiment_instances):
    
    lipizzaner_path = str(pathlib.Path(__file__).parent.absolute()) + "/../lipizzaner/src/main.py"

    experiment_results = []

    # Execute lipizzaner for each experiment
    for instance in experiment_instances:

        start = time.time()

        # Launch clients
        clients_pool = []
        for j in range(grid_size):

            client_command = ["python", lipizzaner_path, "train", "--distributed", "--client"]

            # Important to send output to DEVNULL in order to not populate SLURM file in ClusterUY.
            # Output is logged by Lipizzaner.
            lipizzaner_client = subprocess.Popen(client_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clients_pool.append(lipizzaner_client)
            
            # Important to wait for clients before initializing master
            time.sleep(30)

        # Launch master
        master_command = ["python", lipizzaner_path, "train", "--distributed", "--master", "-f", instance["dir"] + "/main.yml"]

        lipizzaner_master = subprocess.run(master_command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           universal_newlines=True)

        end = time.time()
        wall_clock = end - start

        # Kill clients
        for client in clients_pool:
            client.kill()

        # Search score
        if lipizzaner_master.returncode != 0:
            score = "Non-cero exit code ({})".format(lipizzaner_master.returncode)
        else:
            match = re.search('Best result:.* = \((.*), (.*)\)', lipizzaner_master.stderr)
            if match is None:
                score = "not found"
            else:
                score = match.group(1)

        instance_result = {
            "timestamp":  instance["timestamp"],
            "config_id": instance["config_id"],
            "exec_id": instance["exec_id"],
            "wall_clock": wall_clock,
            "score": score
        }


        experiment_results.append(instance_result)

        result_line = "{}-C{}E{}: {} ({})\n".format(instance_result["timestamp"],
                                                    instance_result["config_id"],
                                                    instance_result["exec_id"],
                                                    instance_result["score"],
                                                    instance_result["wall_clock"])
        print(result_line)                                            

    return experiment_results
