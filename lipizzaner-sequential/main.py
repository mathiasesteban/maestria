import json
import pathlib
import preparer
import os
import time
import runner

from datetime import datetime


def launch():

    start = time.time()

    # Load parameters
    workdir = str(pathlib.Path(__file__).parent.absolute())
    parameters_file = open(str(workdir) + "/config/parameters.json", 'r')
    parameters = json.loads(parameters_file.read())

    # Create output folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_path = workdir + "/../results/sequential-" + timestamp

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # Create summary file
    summary_file = open(experiment_path + "/summary.txt", "w+")

    summary_file.write("Parameters: " + str(parameters) + "\n")

    # Prepare experiment and instance directories
    experiment_instances = preparer.prepare(workdir, experiment_path, parameters)

    summary_file.write("Instances: " + str(len(experiment_instances)) + "\n\n")

    # Run experiment
    experiment_results = runner.run(parameters["grid_size"] ,experiment_instances)

    # Collect results
    result_lines = []
    for result in experiment_results:
        result_line = "{}-C{}E{}: {} ({})\n".format(result["timestamp"],
                                                    result["config_id"],
                                                    result["exec_id"],
                                                    result["score"],
                                                    result["wall_clock"])
        result_lines.append(result_line)

    summary_file.writelines(result_lines)
    summary_file.write("\n")

    end = time.time()
    wall_clock = end - start
    summary_file.writelines("\nTotal wall clock: {}".format(wall_clock))
    summary_file.close()


if __name__ == "__main__":
    launch()

