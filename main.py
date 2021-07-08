from network import Network
from signal_information import SignalInformation
from connection import Connection
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def main():
    avg_latencies = []
    avg_snrs = []
    avg_bitrates = []
    avg_n_connections = []
    avg_total_band = []
    #
    test_conn = 1000
    #
    for i in range(0, test_conn):
         print(i)
         network = Network('nodes.json')
         network.connect()
         connections = []

         nodes_str = ''
         for node in network.nodes:
             nodes_str += node

         for i in range(0, 100):
             while True:
                 src = random.choice(nodes_str)
                 dst = random.choice(nodes_str)
                 if src != dst and network.find_paths(src, dst):
                     break
    #         # print(src + '->' + dst)
             connections.append(Connection(src, dst, 0.001))
    #
         network.stream(connections)
    #
         latencies = []
         snrs = []
         n_conn = 100
    #
         for connection in connections:
             if connection.latency is not None:
                 latencies.append(connection.latency)
                 snrs.append(connection.snr)
             else:
                 n_conn -= 1
    #
         lat_sum = math.fsum(latencies)
         snr_sum = math.fsum(snrs)
    #
         avg_latencies.append(math.fsum(latencies) / len(latencies))
         avg_snrs.append(10 * math.log10(math.fsum(snrs) / len(snrs)))
    #
         print(f'Average latency: {lat_sum / len(latencies)}')
         print(f'Average SNR: {snr_sum / len(snrs)}')
    #
         bit_rates = []
    #
         for connection in connections:
             bit_rates.append(connection.bit_rate)
    #
         # pd.DataFrame(data=bit_rates).plot.hist()
         # plt.show()
    #
         total = math.fsum(bit_rates)
         avg = total / len(bit_rates)
         avg_bitrates.append(avg)
         avg_total_band.append(total)
    #
         print(f'Total: {total} - Average: {avg}')
         print(f'{n_conn} successful connections')
         avg_n_connections.append(n_conn)

    # avg_latencies2 = []
    # avg_snrs2 = []
    # avg_bitrates2 = []
    # avg_n_connections2 = []
    # avg_total_band2 = []
    #
    # for i in range(0, test_conn):
    #     print(i)
    #     network = Network('nodes.json')
    #     network.connect()
    #
    #     for line in network.lines:
    #         network.lines.get(line).noise_figure = 5
    #         network.lines.get(line).beta2 = 0.6e-26
    #
    #     connections = []
    #
    #     nodes_str = ''
    #     for node in network.nodes:
    #         nodes_str += node
    #
    #     for i in range(0, 100):
    #         while True:
    #             src = random.choice(nodes_str)
    #             dst = random.choice(nodes_str)
    #             if src != dst and network.find_paths(src, dst):
    #                 break
    #         # print(src + '->' + dst)
    #         connections.append(Connection(src, dst, 0.001))
    #
    #     network.stream(connections, 'snr')
    #
    #     latencies = []
    #     snrs = []
    #     n_conn = 100
    #
    #     for connection in connections:
    #         if connection.latency is not None:
    #             latencies.append(connection.latency)
    #             snrs.append(connection.snr)
    #         else:
    #             n_conn -= 1
    #
    #     # lat_sum = math.fsum(latencies)
    #     # snr_sum = math.fsum(snrs)
    #
    #     avg_latencies2.append(math.fsum(latencies) / len(latencies))
    #     avg_snrs2.append(10 * math.log10(math.fsum(snrs) / len(snrs)))
    #
    #     # print(f'Average latency: {lat_sum / len(latencies)}')
    #     # print(f'Average SNR: {snr_sum / len(snrs)}')
    #
    #     bit_rates = []
    #
    #     for connection in connections:
    #         bit_rates.append(connection.bit_rate)
    #
    #     # pd.DataFrame(data=bit_rates).plot.hist()
    #     # plt.show()
    #
    #     total = math.fsum(bit_rates)
    #     avg = total / len(bit_rates)
    #     avg_bitrates2.append(avg)
    #     avg_total_band2.append(total)
    #
    #     # print(f'Total: {total} - Average: {avg}')
    #     # print(f'{n_conn} successful connections')
    #     avg_n_connections2.append(n_conn)
    #
    # # df = pd.DataFrame(data={'latency': latencies, 'snr': snrs})
    #
    # df = pd.DataFrame(data={'SNR': avg_snrs, 'SNR - lower beta2, higher NF': avg_snrs2})
    # df.plot(kind='kde', subplots=False, sharex=True, sharey=True)
    #
    print(f'Average latency over {test_conn} connections : {math.fsum(avg_latencies) / len(avg_latencies)}')
    print(f'Average SNR over {test_conn} connections : {math.fsum(avg_snrs) / len(avg_snrs)}')
    print(10 * math.log10(math.fsum(avg_snrs) / len(avg_snrs)))
    print(f'Average total: {math.fsum(avg_total_band) / len(avg_total_band)}')
    print(f'Average bitrate: {math.fsum(avg_bitrates) / len(avg_bitrates)}')
    print(f'Average number of connections: {math.fsum(avg_n_connections) / len(avg_n_connections)}')

    network = Network('nodes.json')
    network.connect()

    nodes = network.nodes
    lines = network.lines
    all_paths = []

    for src in nodes:
        for dst in nodes:
            if src != dst:
                all_paths.append(network.find_paths(src, dst))

    all_paths_ = []

    for paths in all_paths:
        for path in paths:
            all_paths_.append(''.join(path))

    dict_ = {}

    for line in lines:
        dict_.update({line: 0})

    for path in all_paths_:
        for line in lines:
            if path.find(line) != -1:
                x = dict_.get(line)
                dict_.update({line: x + 1})

    plt.bar(dict_.keys(), dict_.values())
    plt.show()

    nodes2 = network.nodes

    G = nx.DiGraph()
    G.add_nodes_from(nodes2.keys())
    for n, p in nodes2.items():
        G.nodes[n]['pos'] = p.position

    for path in all_paths_:
        nx.add_path(G, path)

    plt.figure(figsize=(8, 6))
    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = ['green' if G.degree[e[0]] == 2 else 'red' for e in G.edges]
    nx.draw_networkx(G, pos, arrows=False, with_labels=True, edge_color=edge_colors)
    plt.title("Network")
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
    plt.tick_params(axis='y', which='both', right=True, left=True, labelleft=True)

    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(True)

    plt.show()

    return

    network = Network('nodes.json')
    network_fixed = Network('nodes_full_fixed_rate.json')
    network_flex = Network('nodes_full_flex_rate.json')
    network_shannon = Network('nodes_full_shannon.json')
    network.draw()

    nodes = network.nodes
    all_paths = []  # contains all possible paths between all possible nodes

    network.connect()
    network.generate_traffic()

    network_fixed.connect()
    network_flex.connect()
    network_flex.generate_traffic()

    return
    network_shannon.connect()

    for src in nodes:
        for dst in nodes:
            if src != dst:
                all_paths.append(network.find_paths(src, dst))

    signal_power = 0.001  # 1mW
    results = {}

    for paths in all_paths:
         for path in paths:
             sig_inf = SignalInformation(signal_power, path)
             sig_inf = network.propagate(sig_inf, 0)
             path_str = '->'.join(path)
             results.update(
                 {path_str: [sig_inf.latency, sig_inf.noise_power, 10 * math.log10(signal_power / sig_inf.noise_power)]})

    network.weighted_paths = pd.DataFrame(data=results)
    network.weighted_paths.index = ['Latency (s)', 'Noise power (W)', 'SNR (dB)']
    df = network.weighted_paths
    print(list(df.columns))

    df.columns = ['Latency (s)', 'Noise power (W)', 'SNR (dB)']

    src_node = nodes.get('A').label
    dst_node = nodes.get('B').label

    signal_power = 1
    connections = []
    connections_fixed = []
    connections_flex = []
    connections_shannon = []

    nodes_str = ''
    for node in nodes:
        nodes_str += node

    for i in range(0, 100):
        while True:
            src = random.choice(nodes_str)
            dst = random.choice(nodes_str)
            if src != dst:
                break
        # print(src + '->' + dst)
        connections.append(Connection(src, dst, signal_power))
        connections_fixed.append(Connection(src, dst, signal_power))
        connections_flex.append(Connection(src, dst, signal_power))
        connections_shannon.append(Connection(src, dst, signal_power))

    connections.append(Connection(src_node, dst_node, signal_power))

    network.stream(connections)

    connections_fixed = []
    for i in range(0, 100):
        connections_fixed.append(Connection('A', 'B', signal_power))

    i = 1
    for connection in connections_fixed:
        print(f'{i}:{connection.input}->{connection.output}')
        i += 1

    network_fixed.stream(connections_fixed)
    i = 1
    for connection in connections_fixed:
        print(f'{i}:\t\t{connection.input}->{connection.output}\t{connection.bit_rate}')
        i += 1
    print('\n--------------------------\n')

    network_flex.stream(connections_flex)

    network_shannon.stream(connections_shannon)

    bit_rates_fixed = []
    bit_rates_flex = []
    bit_rates_shannon = []

    for connection in connections_fixed:
        bit_rates_fixed.append(connection.bit_rate)

    for connection in connections_flex:
        bit_rates_flex.append(connection.bit_rate)

    for connection in connections_shannon:
        bit_rates_shannon.append(connection.bit_rate)

    pd.DataFrame(data=bit_rates_fixed).plot.hist()
    plt.savefig('Output_files/bit_rates_fixed.png')
    pd.DataFrame(data=bit_rates_flex).plot.hist()
    plt.savefig('Output_files/bit_rates_flex.png')
    pd.DataFrame(data=bit_rates_shannon).plot.hist()
    plt.savefig('Output_files/bit_rates_shannon.png')
    # plt.show()

    total_fixed = math.fsum(bit_rates_fixed)
    total_flex = math.fsum(bit_rates_flex)
    total_shannon = math.fsum(bit_rates_shannon)
    avg_fixed = total_fixed / len(bit_rates_fixed)
    avg_flex = total_flex / len(bit_rates_flex)
    avg_shannon = total_shannon / len(bit_rates_shannon)

    print(f'Fixed rate:\tTotal: {total_fixed}\tAverage: {avg_fixed}')
    print(f'Flex rate:\tTotal: {total_flex}\tAverage: {avg_flex}')
    print(f'Shannon rate:\tTotal: {total_shannon}\tAverage: {avg_shannon}')


    return

    latencies = []
    snrs = []

    for connection in connections:
        latencies.append(connection.latency)
        snrs.append(connection.snr)

    df = pd.DataFrame(data={'latency': latencies, 'snr': snrs})

    df.plot.kde(subplots=True, sharex=False, sharey=False, title='Best latency')
    df.plot.bar(subplots=True, sharex=False, sharey=False, title='Best latency')
    plt.show()

    # for line in network.lines:
    #     for channel in network.route_space.index:
    #         network.lines.get(line).free(channel)

    network.stream(connections, 'snr')

    latencies = []
    snrs = []

    for connection in connections:
        latencies.append(connection.latency)
        snrs.append(connection.snr)

    df = pd.DataFrame(data={'latency': latencies, 'snr': snrs})
    print(df)

    df.plot.kde(subplots=True, sharex=False, sharey=False, title='Best SNR')
    df.plot.bar(subplots=True, sharex=False, sharey=False, title='Best SNR')
    plt.show()
    print ("ciao")

if __name__ == '__main__':
    main()