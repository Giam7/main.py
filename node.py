from typing import Tuple, List, Dict


class Node:
    def __init__(self, attr_dict: dict):
        """
        :param attr_dict: Dictionary containing Node attributes as values
        """

        self._label = str(attr_dict.get('label'))
        self._position = tuple(attr_dict.get('position'))
        self._connected_nodes = list(attr_dict.get('connected_nodes'))
        self._successive = {}  # dictionary of lines
        self._switching_matrix = None
        self._transceiver = attr_dict.get('transceiver', 'shannon')

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position: Tuple[float, float]):
        self._position = tuple(position)

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @connected_nodes.setter
    def connected_nodes(self, connected_nodes: List[str]):
        self._connected_nodes = list(connected_nodes)

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive: dict):
        self._successive = dict(successive)

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    def update_switching_matrix(self, channel: int, src: str, dst: str, state: int):
        """
        Updates the switching matrix

        :param channel: Number of the channel
        :param src: Source node label
        :param dst: Destination node label
        :param state: 0 is occupied, 1 is free
        """

        self._switching_matrix[src][dst][channel] = state

    def propagate(self, sig_inf, channel: int, connection=False):
        """
        Propagates the signal to the next line and updates the signal path, if the current node is not the last one in the path

        :param sig_inf: SignalInformation or LightPath object
        :param channel: Number of the channel
        :param connection: True if trying to create a Connection over the lines in the path, False by default, optional
        """

        sig_inf.update_path(self)

        if sig_inf.path:
            next_node = sig_inf.path[0]
            for line in self._successive:
                if line.find(next_node) != -1:
                    sig_inf.signal_power = self._successive.get(line).optimized_launch_power()
                    self._successive.get(line).propagate(sig_inf, channel, connection)
