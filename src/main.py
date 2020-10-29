from sys import argv

import numpy as np

from io_helper import read_tsp, normalize, read_obs, normalization
from neuron import generate_network, get_neighborhood, get_route, get_ob_influence
from distance import select_closest, route_distance  # , euclidean_distance
from plot import plot_network, plot_route, update_figure
from gene_tsp import generate_tour
import time


def main():
    # 加载 problem
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        if len(argv) == 1:
            problem = read_tsp("assets/arr.tsp")  # 测试用代码
            obstacle = read_obs("assets/obs.obs")
        else:
            return -1
    else:
        problem = read_tsp(argv[1])  # 读取城市坐标数据
    print("Problem loading completed.")

    # 获得路径结果
    start = time.process_time()
    route_index = som(problem, 100000, 0.8,
                      obstacle)  # from neuron 0 开始的路径 index
    end = time.process_time()
    print('SOM training completed. Running time: %s Seconds' % (end - start))

    # 计算以及评估
    start = time.process_time()
    route = problem.reindex(route_index)
    route.loc[route.shape[0]] = route.iloc[0]  # 末尾添加开头，首尾相连
    plot_route(problem, route, 'diagrams/route.png',
               obstacle=obstacle)  # 画出路径图
    problem = problem.reindex(route_index)  # 对原始的城市进行重新排序
    distance = route_distance(problem)  # 计算城市按照当前路径的距离
    print('Route found of length {}'.format(distance))
    generate_tour(route, length=distance)
    end = time.process_time()
    print('Evaluation completed. Running time: %s Seconds' % (end - start))


def som(problem, iterations, learning_rate=0.8, obstacle=None):
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

    if obstacle is not None:
        cities = problem.copy()[['x', 'y']]
        obs = obstacle.copy()[['x', 'y']]
        norm_ans = normalization(cities, obs)
        cities, obs, dif = norm_ans[0][0], norm_ans[0][1], norm_ans[1]
        obs_n = obs[['x', 'y']].to_numpy()

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8 + obs.shape[0] * 5  # 这里是神经元数目，别误解为人口(population)数目

    # parameters set to observe and evaluate 自己加的
    axes = update_figure()
    gate = 0.01 / dif  # 收敛条件设定，精度的映射
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
        obs_delta = np.apply_along_axis(
            lambda p: get_ob_influence(obs_n, p, (10 + n // 10) * gate),
            axis=1,
            arr=network)
        distances = network[winner_idx] - network
        sep_delta = -np.exp(-distances**2 / (0.434 *
                                             (50 * gate)**2)) * distances
        city_delta = gaussian[:, np.newaxis] * (city - network)
        delta = obs_delta + city_delta + sep_delta
        network += learning_rate * delta
        # Decay the variables
        # 对应了 e^{-t/t0} t0=33332.83
        learning_rate = learning_rate * 0.99997
        # 这部分对应了σ=σ0*e^{-t/t0}, sigma0=n//10 t0=3332.83
        n = n * 0.9997

        # Check for plotting interval
        if not i % 500:  # 1000次画一次图
            plot_network(cities,
                         network,
                         name='diagrams/{:05d}.png'.format(i),
                         axes=axes,
                         obstacle=obs)
            update_figure(axes, clean=True)

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
        if np.linalg.norm(delta, axis=1).mean() < gate:  # 当迭代变化平均值小于设定的精度时停止
            print("Average movement of neuron has reduced to {},".format(gate),
                  "finishing execution at {} iterations".format(i))

            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, network, name='diagrams/final.png', obstacle=obs)

    route = get_route(cities, network)
    return route


if __name__ == '__main__':
    main()
