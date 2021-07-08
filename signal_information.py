from typing import List

from node import Node


class SignalInformation:
    def __init__(self, signal_power: float, path: List[str]):
        """
        :param signal_power: Signal power
        :param path: List of node labels to be crossed
        """

        self._signal_power = signal_power
        self._path = list(path)
        self._noise_power = 0.0  # float
        self._latency = 0.0  # float
        self._isnr = 0.0

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency: float):
        self._latency = latency

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power: float):
        self._signal_power = signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power: float):
        self._noise_power = noise_power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: List[str]):
        self._path = list(path)

    @property
    def isnr(self):
        return self._isnr

    @isnr.setter
    def isnr(self, isnr: float):
        self._isnr = isnr

    def update_path(self, node: Node):
        """
        Remove crossed node from path

        :param node: Node object
        """

        self._path.remove(node.label)

    def increase_latency(self, latency: float):
        """
        Increase latency

        :param latency: Latency value to be added to the signal
        """

        self._latency += latency
