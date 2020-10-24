from sys import argv

import numpy as np

from io_helper import read_tsp, normalize
from neuron import generate_network, get_neighborhood, get_route
from distance import select_closest, route_distance  # , euclidean_distance
from plot import plot_network, plot_route
from gene_tsp import generate_tour


def main():
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        if len(argv) == 1:
            problem = read_tsp("assets/arr.tsp")  # 测试用代码
        else:
            return -1
    else:
        problem = read_tsp(argv[1])  # 读取城市坐标数据

    route = som(problem, 100000)  # from neuron 0 开始的路径 index

    problem = problem.reindex(route)  # 对原始的城市进行重新排序

    distance = route_distance(problem)  # 计算城市按照当前路径的距离

    print('Route found of length {}'.format(distance))
    generate_tour(route, length=distance)


def som(problem, iterations, learning_rate=0.8):
    """
    problem: [DataFrame] ['city', 'y', 'x']
    iterations: [int] the max iteration times
    learning rate: [float] the original learning rate, will decay
    return: [index] route

    Solve the TSP using a Self-Organizing Map.
    """

    # Obtain the normalized set of cities (w/ coord in [0,1])
    # copy one so the later process won't influence the original data
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8  # 这里是神经元数目，别误解为人口(population)数目

    # Generate an adequate network of neurons:
    network = generate_network(n)  # 2列矩阵
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            # "\r"回车，将光标移到本行开头，大概就是覆盖了吧
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
        # Choose a random city
        # 随机选取某一行 cities.sample(1)
        # [['x', 'y']] 筛出这两列
        # DataFrame.values --> numpy.ndarray
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])
        # Update the network's weights (closer to the city)
        # newaxis is the alias(别名) for None 为了调整array的结构，否则无法参与运算
        # 具体应该是broadcast相关原理
        network += gaussian[:, np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        # 对应了 e^{-t/t0} t0=33332.83
        learning_rate = learning_rate * 0.99997
        # 这部分对应了σ=σ0*e^{-t/t0}, sigma0=n//10 t0=3332.83
        n = n * 0.9997

        # Check for plotting interval
        if not i % 1000:  # 1000次画一次图
            plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, network, name='diagrams/final.png')

    route = get_route(cities, network)
    plot_route(cities, route, 'diagrams/route.png')
    return route


if __name__ == '__main__':
    main()
