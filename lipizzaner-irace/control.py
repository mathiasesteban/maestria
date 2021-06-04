import subprocess as sp
import os
import pickle


class ExecutionControl:

    def __init__(self, pid, min_port, max_port, grid_size, required_mem):

        self.execution_stack = []
        self.available_gpus = []
        self.gpus_total_memory = []

        self.ports_windows = []
        self.available_ports_windows = []
        self.results_stack = []
        self.required_mem = required_mem * 1.10

        self.logs = []

        # Find port windows based on grid size
        if grid_size == 1:
            for port in range(min_port, max_port + 1):
                self.ports_windows.append(str(port))
                self.available_ports_windows.append(0)
        else:
            n_windows = (min_port - max_port) // grid_size

            for i in range(0, n_windows):

                lower_port = min_port + (i * grid_size)
                upper_port = min_port + (i * grid_size) + grid_size

                self.ports_windows.append("{}-{}".format(lower_port, upper_port))
                self.available_ports_windows.append(0)

        # Find available gpus
        gpu_count = self.get_available_gpus()
        for i in range(0, gpu_count):
            self.available_gpus.append([])
            self.gpus_total_memory.append(self.get_gpus_total_memory())

        self.logs.append("Process {} created controller.".format(pid))

    def add_to_queue(self, pid):
        self.execution_stack.append(pid)
        self.logs.append("Process {} added to queue.".format(pid))

    def is_my_turn(self, pid):
        try:
            can_start = True
            resources = {"port_windows_idx": None, "ports": None, "gpu": None}

            # Check if process is the next in the queue
            if self.execution_stack[0] != pid:
                self.logs.append("Process {} denied to run (is not the next in the queue).".format(pid))
                can_start = False

            # Check if there are available portes
            if can_start:
                port_window_index = get_index(self.available_ports_windows, 0)

                if port_window_index is not None:
                    resources["port_windows_idx"] = port_window_index
                    resources["ports"] = self.ports_windows[port_window_index]
                else:
                    can_start = False
                    self.logs.append("Process {} denied to run (no available ports) .".format(pid))

            # Check if there are available gpus
            available_gpu = False
            if can_start:
                for idx, gpu in enumerate(self.available_gpus):

                    is_available_memory = self.get_gpu_memory()[idx] > self.required_mem
                    no_race_condition = (len(self.available_gpus[idx]) + 1) * self.required_mem < self.gpus_total_memory[idx]

                    if is_available_memory and no_race_condition:
                        available_gpu = True
                        resources["gpu"] = idx
                        break

                if not available_gpu:
                    can_start = False
                    self.logs.append("Process {} denied to run (no available gpus).".format(pid))

            # Finally check if process is allowed to run.
            if can_start:
                self.logs.append("Process {} allowed to run (port={}, gpu={}).".format(pid, resources["ports"], resources["gpu"]))
                self.available_ports_windows[resources["port_windows_idx"]] = pid
                self.available_gpus[resources["gpu"]].append(pid)
                self.execution_stack.pop(0)
                return resources
            else:
                return None
        except Exception as e:
            self.logs.append("ERROR: {}".format(e))
            raise Exception("Execution controller exception")

    def free_resources(self, pid, score, time):
        self.logs.append("Process {} finished.".format(pid))

        # Free port resources
        port_window_index = get_index(self.available_ports_windows, pid)
        self.available_ports_windows[port_window_index] = 0

        # Free gpus
        for gpu in self.available_gpus:
            process_index = get_index(gpu, pid)

            if process_index is not None:
                gpu.pop(process_index)

        result = {"pid": pid, "score": score, "time": time}
        self.results_stack.append(result)

    def get_available_gpus(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        command = "nvidia-smi -L"
        gpu_list = _output_to_list(sp.check_output(command.split()))
        return len(gpu_list)

    def get_gpus_total_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        gpu_list = _output_to_list(sp.check_output(command.split()))
        gpu_list[1].split()
        return int(gpu_list[1].split()[0])

    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values

    def print_report(self):

        print("--------------------------------------------")
        print("Established port windows: {}".format(self.ports_windows))
        print("Port resources state: {}".format(self.available_ports_windows))
        print("GPU resources state: {}".format(self.available_gpus))
        print("GPU total memory: {}".format(self.gpus_total_memory))
        print("--------------------------------------------")
        print("Execution stack: {}".format(self.execution_stack))
        print("Results stack: {}".format(self.results_stack))
        print("--------------------------------------------")
        print("LOG lines")
        print("--------------------------------------------")

        for line in self.logs:
            print(line)


def get_index(my_list, my_element):
    try:
        return my_list.index(my_element)
    except ValueError:
        return None


if __name__ == "__main__":

    execution_control_file_path = '../results/irace/control.pkl'

    if os.path.isfile(execution_control_file_path):

        execution_control_file = open(execution_control_file_path, 'rb')
        execution_control = pickle.load(execution_control_file)

    else:
        pid = 500
        min_port = 5000
        max_port = 5000
        grid_size = 1
        required_mem = 3000

        execution_control = ExecutionControl(pid, min_port, max_port, grid_size, required_mem)

    execution_control.print_report()
