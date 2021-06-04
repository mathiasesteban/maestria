from distribution.client_api import ClientAPI
from distribution.client_environment import ClientEnvironment
from helpers.network_helpers import is_port_open
from helpers.configuration_container import ConfigurationContainer

DEFAULT_CLIENT_PORT = 5000
MAX_CLIENT_PORT = 5500


class LipizzanerClient:
    def run(self):


        # Quickfix to launch multiple LIpizzaner experiments in the same
        cc = ConfigurationContainer.instance()

        print("******************************")
        print("settings AT: {}".format(cc.settings))
        print("******************************")

        ports_setting = cc.settings['general']['distribution']['client_nodes'][0]['port']

        # Single client grid the port settings is a single int
        if type(ports_setting) is int:
            DEFAULT_CLIENT_PORT = ports_setting
            MAX_CLIENT_PORT = ports_setting
        # In square grids ports is a string XXXX-XXXX
        else:
            ports = ports_setting.split('-')
            DEFAULT_CLIENT_PORT = int(ports[0])
            MAX_CLIENT_PORT = int(ports[len(ports) - 1])

        # Set the lowest to start search
        port = DEFAULT_CLIENT_PORT

        while not is_port_open(port):

            print("******************************")
            print("TRYING AT: {}".format(port))
            print("******************************")


            if port == MAX_CLIENT_PORT:
                raise IOError('No free port between {} and {} available.'.format(DEFAULT_CLIENT_PORT, MAX_CLIENT_PORT))

            port += 1

        ClientEnvironment.port = port


        print("******************************")
        print("LAUNCHING AT: {}".format(port))
        print("******************************")

        ClientAPI().listen(port)
