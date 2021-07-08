from typing import List

from signal_information import SignalInformation


class LightPath(SignalInformation):
    def __init__(self, signal_power: float, path: List[str]):
        super().__init__(signal_power, path)
        self._channel = None
        self._Rs = 32  # symbol rate in GHz
        self._df = 50  # frequency spacing between channels in GHz

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, channel):
        self._channel = channel

    @property
    def Rs(self):
        return self._Rs

    @Rs.setter
    def Rs(self, Rs: float):
        self._Rs = Rs

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: float):
        self._df = df
