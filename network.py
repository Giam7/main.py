import json
import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from lightpath import LightPath
from signal_information import SignalInformation
from node import Node
from line import Line
from connection import Connection
from scipy import special


class Network:
    def __init__(self, json_file: str):
        """
        :param json_file: String containing json file path
        """

        with open(json_file, "r") as f:
            tmp_dict = json.load(f)

        self._nodes = {}

        for node in tmp_dict:
            tmp_dict.get(node).update({'label': node})
            tmp_node = Node(tmp_dict.get(node))
            tmp_node.label = node
            self._nodes.update({node: tmp_node})

        self._lines = {}

        for node in self._nodes:
            for connected_node in self._nodes.get(node).connected_nodes:
                tmp_length = math.dist(self._nodes.get(node).position, self._nodes.get(connected_node).position)
                tmp_line = Line(self._nodes.get(node).label + connected_node, tmp_length)
                tmp_line.n_amplifiers = int(tmp_line.length/80000)  # amplifier needed every 80 km
                self._lines.update({tmp_line.label: tmp_line})

        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: dict):
        self._nodes = nodes

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines: dict):
        self._lines = lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @weighted_paths.setter
    def weighted_paths(self, weighted_paths: pd.DataFrame):
        self._weighted_paths = weighted_paths

    @property
    def route_space(self):
        return self._route_space

    def connect(self):
        """
        Sets the successive attributes of all the network elements as dictionaries, updates route_space and switching_matrix
        """

        for node in self._nodes:
            tmp_dict_nodes = {}
            for line in self._lines:
                tmp_dict_lines = {}
                if line.find(node) == 1:  # if a line contains a node as destination, update succ dictionary in the line
                    tmp_dict_lines.update({node: self._nodes.get(node)})
                    self._lines.get(line).successive = tmp_dict_lines
                elif line.find(node) == 0:  # if a line contains a node as source, update succ dictionary in the node
                    tmp_dict_nodes.update({line: self._lines.get(line)})

            self._nodes.get(node).successive = tmp_dict_nodes

        all_paths = []

        for src in self._nodes:
            for dst in self._nodes:
                if src != dst:
                    all_paths.append(self.find_paths(src, dst))

        signal_power = 0.001
        results = {}

        for paths in all_paths:
            for path in paths:
                sig_inf = SignalInformation(signal_power, path)
                sig_inf = self.propagate(sig_inf, 0)
                path_str = '->'.join(path)
                results.update(
                    {path_str: [sig_inf.latency, sig_inf.noise_power,
                                10 * math.log10(1/sig_inf.isnr)]})

        self.weighted_paths = pd.DataFrame(data=results)
        self.weighted_paths.index = ['Latency (s)', 'Noise power (W)', 'SNR (dB)']

        channels = {}
        for i in range(0, 10):
            channels.update({i: 'free'})

        paths = results.keys()
        route_space = {}

        for path in paths:
            route_space.update({path: channels})

        self._route_space = pd.DataFrame(data=route_space)

        ones = []
        zeros = []
        for i in range(0, 10):
            ones.append(1)
            zeros.append(0)

        switching_matrix = {}

        for node in self._nodes:
            for node2 in self._nodes:
                if node2 != node:
                    switching_matrix.update({node2: {}})
                    for node3 in self._nodes:
                        if node not in self._nodes.get(node2).connected_nodes or node2 == node3 or node3 not in self._nodes.get(node).connected_nodes:
                            switching_matrix.get(node2).update({node3: np.array(zeros)})
                        elif node3 != node:
                            switching_matrix.get(node2).update({node3: np.array(ones)})
                    switching_matrix.get(node2).pop(node)

            self._nodes.get(node).switching_matrix = switching_matrix
            switching_matrix = {}

    def find_paths(self, node1_label: str, node2_label: str):
        """
        Finds all paths that connect the two nodes

        :param node1_label: Label of first node
        :param node2_label: Label of second node
        :return: list of paths
        :rtype: list[list[str]]
        """

        node1 = self._nodes.get(node1_label)
        paths = []

        tmp_path = [node1_label]
        self.__find_path(node1, node2_label, paths, tmp_path)

        return paths

    def __find_path(self, node: Node, dest_node_label: str, paths: List[List[str]],
                    tmp_path: List[str]):  # internal recursive function to find all paths
        if dest_node_label in node.connected_nodes:
            tmp_path.append(dest_node_label)
            paths.append(tmp_path.copy())
            tmp_path.remove(dest_node_label)  # remove last element, path already saved

        for node_ in node.connected_nodes:  # check if there are other longer paths remaining
            if node_ not in tmp_path and node_ != dest_node_label:
                tmp_path.append(node_)
                self.__find_path(self._nodes.get(node_), dest_node_label, paths, tmp_path)
                tmp_path.remove(node_)

    def propagate(self, sig_inf: SignalInformation, channel: int = 0, connection=False):
        """
        Propagates the signal through the path specified in it

        :param sig_inf: SignalInformation object
        :param connection: True if trying to create a Connection over the lines in the path, False by default, optional
        :param channel: Number of the channel
        :return: updated SignalInformation object
        :rtype: SignalInformation
        """

        if isinstance(sig_inf, LightPath):
            sig_inf.channel = channel
            sig_path = sig_inf.path.copy()
            self._nodes.get(sig_inf.path[0]).propagate(sig_inf, channel, connection)
            if connection:
                self._route_space.at[channel, '->'.join(sig_path)] = 'occupied'
                for i in range(1, len(sig_path) - 1):
                    node = sig_path[i]
                    self._nodes.get(node).update_switching_matrix(channel, sig_path[i - 1], sig_path[i + 1], 0)
                    for path in self._route_space.columns:
                        for j in range(0, len(sig_path) - 1):
                            if '->'.join([sig_path[j], sig_path[j + 1]]) in path:
                                self._route_space.at[channel, path] = 'occupied'

        else:
            self._nodes.get(sig_inf.path[0]).propagate(sig_inf, channel, connection)

        return sig_inf

    def draw(self):
        """
        Draws the network
        """
        nodes_x = {}
        nodes_y = {}

        for node in self._nodes.values():
            nodes_x.update({node.label: node.position[0]})
            nodes_y.update({node.label: node.position[1]})

        fig = plt.figure()
        ax = fig.add_subplot()

        x = np.array(list(nodes_x.values()))
        y = np.array(list(nodes_y.values()))

        lines_x = {}
        lines_y = {}

        for line in self._lines:
            lines_x.update({line: (nodes_x.get(line[0]), nodes_x.get(line[1]))})
            lines_y.update({line: (nodes_y.get(line[0]), nodes_y.get(line[1]))})

        for line in self._lines:
            xx = np.array(lines_x.get(line))
            yy = np.array(lines_y.get(line))
            plt.plot(xx, yy, linestyle='-', linewidth=2)

        plt.plot(x, y, marker='o', ls='', c='lightblue', ms=20)  # plot nodes

        nodes_pos = {}

        for node in self._nodes.values():
            nodes_pos.update({node.position: node.label})

        for xy in zip(x, y):
            ax.annotate(f'{nodes_pos.get(xy)}', fontweight='bold', fontsize='large', ma='left', va='center', c='black',
                        xy=xy, xytext=(xy[0] - 7000, xy[1]), textcoords='data')

        plt.title('Network')

        plt.show()

    def find_best_snr(self, src_node: str, dst_node: str, channel: int = -1):
        """
        Finds path with highest SNR

        :param src_node: Source Node label
        :param dst_node: Destination Node label
        :param channel: Number of the channel
        :return: path with highest SNR
        :rtype: str
        """

        best_snr = 0
        best_snr_path = ''

        # if self._weighted_paths.empty:
        #     print("Paths not defined")
        # else:
        #     tmp_paths = self._weighted_paths.to_dict()
        #     for path in tmp_paths:
        #         if path[0] == src_node and path[-1] == dst_node:
        #             snr = tmp_paths.get(path).get('SNR (dB)')
        #             if snr > best_snr:
        #                 best_snr = snr
        #                 best_snr_path = path

        signal_power = 0.001

        paths = self.find_paths(src_node, dst_node)

        for path in paths:
            path_str = '->'.join(path)
            sig_inf = SignalInformation(signal_power, path)
            if channel != -1:
                sig_inf = self.propagate(sig_inf, channel)
            else:
                sig_inf = self.propagate(sig_inf)
            if sig_inf.latency != 0 and sig_inf.noise_power != 0:
                # snr = math.log10(signal_power / sig_inf.noise_power)
                snr = 1/sig_inf.isnr
                if snr > best_snr:
                    best_snr = snr
                    best_snr_path = path_str

        return best_snr_path, best_snr

    def find_best_latency(self, src_node: str, dst_node: str, channel: int = -1):
        """
        Finds path with lowest latency

        :param src_node: Source Node label
        :param dst_node: Destination Node label
        :param channel: Number of the channel
        :return: path with lowest latency
        :rtype: str
        """

        best_lat = math.inf
        best_lat_path = ''

        # if self._weighted_paths.empty:
        #     print("Paths not defined")
        # else:
        #     tmp_paths = self._weighted_paths.to_dict()
        #     for path in tmp_paths:
        #         if path[0] == src_node and path[-1] == dst_node:
        #             lat = tmp_paths.get(path).get('Latency (s)')
        #             if lat < best_lat:
        #                 best_lat = lat
        #                 best_lat_path = path

        signal_power = 0.001

        paths = self.find_paths(src_node, dst_node)

        for path in paths:
            path_str = '->'.join(path)
            sig_inf = SignalInformation(signal_power, path)
            if channel != -1:
                sig_inf = self.propagate(sig_inf, channel)
            else:
                sig_inf = self.propagate(sig_inf)
            if sig_inf.latency != 0 and sig_inf.noise_power != 0:
                if sig_inf.latency < best_lat:
                    best_lat = sig_inf.latency
                    best_lat_path = path_str

        return best_lat_path, best_lat

    def stream(self, connections: List[Connection], label: str = 'latency'):
        """
        Sets latency and SNR for each connection in the list

        :param connections: list of Connection objects
        :param label: 'snr' or 'latency'
        """

        if label not in ['snr', 'latency']:
            print("Invalid label")
            return

        for connection in connections:
            best_snr = 0
            best_latency = math.inf
            best_channel = None
            best_path = ''

            if label == 'snr':
                for channel in self._route_space.index:
                    (path, snr) = self.find_best_snr(self._nodes.get(connection.input).label,
                                                     self._nodes.get(connection.output).label, channel)
                    if snr > best_snr:
                        best_path = path
                        best_channel = channel
                        best_snr = snr
            else:
                for channel in self._route_space.index:
                    (path, latency) = self.find_best_latency(self._nodes.get(connection.input).label,
                                                             self._nodes.get(connection.output).label, channel)
                    if latency < best_latency:
                        best_path = path
                        best_channel = channel
                        best_latency = latency

            if best_path != '':
                sig_inf = LightPath(connection.signal_power, best_path.split('->'))
                bit_rate = self.calculate_bit_rate(sig_inf, self._nodes.get(best_path[0]).transceiver, best_channel)
                if bit_rate != 0:
                    # sig_inf = LightPath(connection.signal_power, best_path.split('->'))
                    self.propagate(sig_inf, best_channel, True)
                    # connection.snr = 10 * math.log10(connection.signal_power / sig_inf.noise_power)
                    connection.snr = 1/sig_inf.isnr
                    connection.latency = sig_inf.latency
                    connection.bit_rate = bit_rate
                else:
                    connection.snr = 0.0
                    connection.latency = None
            else:
                connection.snr = 0.0
                connection.latency = None

    def calculate_bit_rate(self, lightpath: LightPath, strategy: str, channel: int = 0):
        """
        Evaluates the bitrate supported by the given path

        :param path: Path
        :param strategy: Transceiver technology
        :param channel: Channel
        :return: Maximum bitrate supported by the path in Gbps
        :rtype: int
        """

        signal_power = 0.001
        sig_inf = SignalInformation(signal_power, lightpath.path)
        sig_inf = self.propagate(sig_inf, channel)

        if sig_inf.latency != 0 and sig_inf.noise_power != 0:
            # snr = sig_inf.signal_power / sig_inf.noise_power
            snr = 1/sig_inf.isnr
        else:
            return 0.0

        BERt = 1e-3  # bit error rate
        Rs = lightpath.Rs  # symbol rate of the lightpath in GHz
        Bn = 12.5  # noise bandwidth in GHz
        Rb = 0

        if strategy == 'fixed_rate':
            if snr >= 2 * ((special.erfcinv(2 * BERt)) ** 2) * Rs / Bn:
                Rb = 100
            else:
                Rb = 0
        elif strategy == 'flex_rate':
            if snr < 2 * ((special.erfcinv(2 * BERt)) ** 2) * Rs / Bn:
                Rb = 0
            elif snr < 14 / 3 * ((special.erfcinv(3 / 2 * BERt)) ** 2) * Rs / Bn:
                Rb = 100
            elif snr < 10 * ((special.erfcinv(8 / 3 * BERt)) ** 2) * Rs / Bn:
                Rb = 200
            else:
                Rb = 400
        elif strategy == 'shannon':
            Rb = 2 * Rs * math.log2(1 + snr * Bn / Rs)

        return Rb

