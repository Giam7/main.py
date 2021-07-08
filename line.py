import math
from typing import Dict, List
from scipy import constants


class Line:
    def __init__(self, label: str, length: float):
        """
        :param label: Line label
        :param length: Line length in meters
        """

        self._label = str(label)
        self._length = float(length)
        self._successive = {}  # dictionary containing one node
        self._state = []  # each state must be 'free' or 'occupied'
        self._n_amplifiers = 0
        self._gain = 16  # dB
        self._noise_figure = 3  # dB
        # self._noise_figure = 5  # dB
        self._alpha_dB = 0.2e-3  # dB/m
        self._beta2 = 2.13e-26  # (m*Hz^2)^(-1)
        # self._beta2 = 0.6e-26  # (m*Hz^2)^(-1)
        self._gamma = 1.27e-3  # (W*m)^(-1)

        for i in range(0, 10):
            self._state.append('free')

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length: float):
        self._length = length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive: dict):
        self._successive = successive

    @property
    def state(self):
        return self._state

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @n_amplifiers.setter
    def n_amplifiers(self, n_amplifiers: int):
        self._n_amplifiers = n_amplifiers

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain: float):
        self._gain = gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, noise_figure: float):
        self._noise_figure = noise_figure

    @state.setter
    def state(self, state: List[str]):
        if state in ['free', 'occupied']:
            self._state = state
        else:
            print('State not allowed')

    @property
    def alpha_dB(self):
        return self._alpha_dB

    @alpha_dB.setter
    def alpha_dB(self, alpha_dB: float):
        self._alpha_dB = alpha_dB

    @property
    def beta2(self):
        return self.beta2

    @beta2.setter
    def beta2(self, beta2: float):
        self._beta2 = beta2

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
        self._gamma = gamma

    def is_free(self, channel: int):
        """
        Tells if the selected channel is free or not

        :param channel: Number of the channel
        :return: True or False
        :rtype: bool
        """

        if self._state[channel] == 'free':
            return True
        else:
            return False

    def occupy(self, channel: int):
        """
        Occupies the selected channel

        :param channel: Number of the channel
        """
        self._state[channel] = 'occupied'

    def free(self, channel: int):
        """
        Frees the selected channel

        :param channel: Number of the channel
        """

        self._state[channel] = 'free'

    def latency_generation(self):
        """
        Generates latency added by the line

        :return: latency
        :rtype: float
        """

        c = 2.998e8
        v = float(2 / 3 * c)
        return self._length / v

    def noise_generation(self, signal_power: float):
        """
        Generates noise added by the line to the signal

        :param signal_power: Signal power
        :return: generated noise
        :rtype: float
        """

        # return 1e-9 * signal_power * self._length
        return self.ase_generation() + self.nli_generation(signal_power)

    def ase_generation(self):
        """
        Evaluates the amount of ASE generated by the amplifiers
        :return: ASE in linear units
        :rtype: float
        """
        N = self._n_amplifiers
        h = constants.h  # Planck constant
        f = 193.414e12  # C-band center
        Bn = 12.5e9  # noise bandwidth
        NF = 10 ** (self._noise_figure / 10)
        G = 10 ** (self._gain / 10)

        x = N * (h * f * Bn * NF * (G - 1))

        return x

    def nli_generation(self, power: float):
        """
        Evaluates the total amount of nonlinear interference noise

        :param power: Signal power
        :return: NLI in linear units
        :rtype: float
        """

        alpha = self._alpha_dB / (20 * math.log10(math.e))
        Rs = 32e9
        df = 50e9
        Nch = 10
        N = self._n_amplifiers  # N span
        eta_nli = 16 / (27 * math.pi) * math.log(
            (math.pi ** 2) / 2 * abs(self._beta2) * (Rs ** 2) / alpha * Nch ** (2 * Rs / df)) * (self._gamma ** 2) / (
                              4 * alpha * self._beta2) * 1 / (Rs ** 3)
        Bn = 12.5e9

        return power ** 3 * eta_nli * N * Bn

    def optimized_launch_power(self):
        """
        Returns optimal power to maximize GNSR

        :return: Optimized launch power
        :rtype: float
        """

        alpha = self._alpha_dB / (20 * math.log10(math.e))
        # loss = 10 ** (-self._alpha_dB * self._length / 10)
        loss = 1
        h = constants.h  # Planck constant
        f = 193.414e12  # C-band center
        Bn = 12.5e9
        Rs = 32e9
        df = 50e9
        Nch = 10
        eta_nli = 16 / (27 * math.pi) * math.log(
            (math.pi ** 2) / 2 * abs(self._beta2) * (Rs ** 2) / alpha * Nch ** (2 * Rs / df)) * (self._gamma ** 2) / (
                          4 * alpha * self._beta2) * 1 / (Rs ** 3)
        NF = 10 ** (self._noise_figure / 10)

        # return ((NF*loss*h*f*Bn)/(2*Bn*eta_nli))**(1.0/3.0)
        ase = self.ase_generation()/self.n_amplifiers
        return (ase / (2 * eta_nli * Bn)) ** (1 / 3)

    def propagate(self, sig_inf, channel: int, connection=False):
        """
        Propagates the signal to the next node, adding noise and latency to the signal

        :param sig_inf: SignalInformation or LightPath object
        :param channel: Number of the channel
        :param connection: True if trying to create a Connection over the Line, False by default, optional
        """

        if self.is_free(channel):
            sig_inf.noise_power += self.noise_generation(sig_inf.signal_power)
            sig_inf.increase_latency(self.latency_generation())
            sig_inf.isnr += 1/(sig_inf.signal_power/sig_inf.noise_power)
            if connection:
                self.occupy(channel)
            self._successive.get(sig_inf.path[0]).propagate(sig_inf, channel, connection)
        else:
            sig_inf.noise_power = 0
            sig_inf.latency = 0
            sig_inf.isnr = math.inf
