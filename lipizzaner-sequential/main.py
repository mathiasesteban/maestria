import json
import pathlib
import preparer
import os
import time
import runner

from datetime import datetime


def launch():

    start = time.time()

    # Load config
    workdir = str(pathlib.Path(__file__).parent.absolute())
    config_file = open(str(workdir) + "/parameters.json", 'r')
    config = json.loads(config_file.read())

    # Create output folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_path = workdir + "/outputs/" + timestamp

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # Create summary file
    summary_file = open(experiment_path + "/summary.txt", "w+")
    summary_file.write("Lipizzaner: " + config["lipizzaner_path"] + "\n")
    summary_file.write("Grid size: " + str(config["grid_size"]) + "\n")
    summary_file.write("Executions: " + str(config["n_executions"]) + "\n")

    # Prepare experiment and instance directories
    experiment_instances = preparer.prepare(workdir, experiment_path, config)

    summary_file.write("Instances: " + str(len(experiment_instances)) + "\n\n")

    # Run experiment
    experiment_results = runner.run(config, experiment_instances)

    # Collect results
    result_lines = []
    for result in experiment_results:
        result_line = "{}-C{}E{}: {} ({})\n".format(result["timestamp"],
                                                    result["config"],
                                                    result["exec"],
                                                    result["fid"],
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

