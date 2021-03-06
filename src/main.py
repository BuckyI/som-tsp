import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from io_helper import read_tsp, normalize, read_obs, normalization, get_gif, save_info, read_fbz, data_io
from neuron import generate_network, get_neighborhood, get_route, get_ob_influences, get_route_vector, ver_vec, sepaprate_node, is_point_in_polygon, sep_and_close_nodes, cluster
from distance import select_closest, route_distance  # , euclidean_distance
from plot_data import plot_network, plot_route, update_figure
from gene_tsp import generate_tour
import time
import logging
import random
from sklearn.cluster import KMeans


def get_parser():
    """
    从命令行读取参数
    return argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="TSP arguments",
                                     usage="python src/main.py <filename>.tsp")
    parser.add_argument('-t',
                        '--target',
                        metavar="<filename>.tsp",
                        required=True,
                        help="init tsp targets")
    parser.add_argument('-o',
                        '--obstacle',
                        metavar="<filename>.obs",
                        help="load obstacles")
    parser.add_argument('-f',
                        '--forbidzone',
                        metavar="<filename>.fbz",
                        help="load forbidden zone")
    return parser.parse_args()


def main():
    # 本次程序运行相关配置设定
    time_id = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 标识本次运行
    data_path = "assets/" + time_id + "/"  # 作为运行数据存储的路径
    os.mkdir(data_path)  # 建立文件夹

    st = logging.StreamHandler()
    st.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    st.setLevel(logging.INFO)  # 屏幕只显示 INFO 以上
    st.addFilter(logging.Filter('root'))
    fh = logging.FileHandler(data_path + 'log.log', 'a')
    fh.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(funcName)s: %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S %a'))
    fh.addFilter(logging.Filter('root'))
    logging.basicConfig(level=logging.DEBUG, handlers=[st, fh])

    start_time = time.process_time()

    # 加载 problem
    arg = get_parser()
    target = read_tsp(arg.target)  # 读取城市坐标数据
    obstacle = read_obs(arg.obstacle) if arg.obstacle is not None else None
    fbzs = read_fbz(arg.forbidzone) if arg.forbidzone is not None else None
    logging.info("Problem loading completed.")

    # 获得路径结果
    # distance = som(target, 100000, 0.8, obstacle, fbzs,
    #                data_path)  # from neuron 0 开始的路径 index
    distance = multi_som(target, 100000, 0.8, obstacle, fbzs, data_path)

    run_time = time.process_time() - start_time
    logging.info('SOM training completed. Running time: %s Seconds', run_time)

    # 生成相关文件
    get_gif(data_path)
    save_info(
        data_path,
        target=arg.target,
        obstacle=arg.obstacle,
        run_time=run_time,
        distance=distance,
    )


def som(target,
        iterations,
        learning_rate=0.8,
        obstacle=None,
        fbzs=None,
        data_path="assets/"):
    """
    target: [DataFrame] ['city', 'y', 'x']
    iterations: [int] the max iteration times
    learning rate: [float] the original learning rate, will decay
    obstacle: [DataFrame] ['obs' 'y' 'x']
    data_path: [str] 迭代过程中以文件形式保存的数据的路径
    
    return: [index] route

    Solve the TSP using a Self-Organizing Map.
    """

    # Obtain the normalized set of cities (w/ coord in [0,1])
    # copy one so the later process won't influence the original data
    cities = target.copy()[['x', 'y']]
    obs = obstacle.copy()[['x', 'y']] if obstacle is not None else None

    norm_ans = normalization(fbzs, cities, obs)
    cities, obs, span, fbzs = norm_ans["result"][0], norm_ans["result"][
        1], norm_ans["dif"], norm_ans["fbzs"]
    obs = obs[['x', 'y']].to_numpy()
    targets = cities[['x', 'y']].to_numpy()
    # The population size is 8 times the number of cities
    n = targets.shape[0] * 8  # 这里是神经元数目，别误解为人口(population)数目
    n = n + obs.shape[0] * 2 if obstacle is not None else n
    n = n + len(fbzs) * 2 if obstacle is not None else n

    # parameters set to observe and evaluate 自己加的
    axes = update_figure()
    old_delta, old_network = [], 0  # 用来判断变化大小的收敛变量
    gate = 1 / span  # 收敛条件设定，精度的映射
    obs_size = 4 * gate
    # Generate an adequate network of neurons:
    network = generate_network(n)  # 2列矩阵
    logging.info('Network of %s neurons created. Starting iterations:', n)

    for i in range(iterations):
        if not i % 100:
            # "\r"回车，将光标移到本行开头，大概就是覆盖了吧
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")

        route_dir_vec = get_route_vector(network, d=0, t=1)  # 当前路径下的顺时针,出发方向向量

        # Choose a random city
        # DataFrame.values --> numpy.ndarray
        # city = cities.sample(1)[['x', 'y']].values
        city = random.choice(targets)  # 随机选取一个目标
        winner_idx = select_closest(network, city)
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])
        city_delta = gaussian[:, np.newaxis] * (city - network)
        network += learning_rate * city_delta

        # choose a random obstacle
        # if obs is None:
        #     obs_delta = 0
        # else:
        #     # obs_influence = ver_vec(np.roll(route_dir_vec, 1, axis=0),
        #     #                         get_ob_influences(network, obs, obs_size))
        #     obs_delta = get_ob_influences(network, obs, obs_size)
        #     network += learning_rate * obs_delta

        # adjust the forbidden area
        # if fbzs is not None:
        #     fbzs_delta = np.zeros(network.shape)
        #     for fbz in fbzs:
        #         for index, node in enumerate(network):
        #             if is_point_in_polygon(node, fbz) == 1:
        #                 ver_dist_v = ver_vec(get_route_vector(fbz), fbz - node)
        #                 # 计算 node to 边界的距离并找到最小值的位置
        #                 ver_dist = np.linalg.norm(ver_dist_v, axis=1)
        #                 closest = ver_dist.argmin()

        #                 # update delta
        #                 fbzs_delta[index] += ver_dist_v[closest]
        #             # 这里可以添加安全距离 / ver_dist[closest] * 1
        # a = np.linalg.norm(fbzs_delta, axis=1)
        # fbzs_delta = ver_vec(np.roll(route_dir_vec, 1, axis=0),
        #                      fbzs_delta)  # 垂直方向影响
        # b = np.linalg.norm(fbzs_delta, axis=1)
        # fbzs_delta[b != 0] *= (a[b != 0] / b[b != 0])[:, np.newaxis]
        # network += fbzs_delta

        # Update the network's weights (closer to the city)
        # delta = city_delta + obs_delta
        # network += learning_rate * delta

        # 修正结点分布,使之间隔更加均匀
        # network = sepaprate_node(network)
        winner_indices = np.apply_along_axis(
            func1d=lambda t: select_closest(network, t),
            axis=1,
            arr=targets,
        )  # 胜者不改变
        network = sep_and_close_nodes(
            network,
            decay=learning_rate,
            targets=targets,
            obstacle=obs,  # 圆形障碍物
            obs_size=obs_size,  # 障碍物半径
            fbzs=fbzs,  # 不规则障碍物
            gate=gate,  # 最大更新步长
            winner_indices=winner_indices,
        )
        # Decay the variables
        # 学习率更新 对应了 e^{-t/t0} t0=33332.83
        learning_rate = learning_rate * 0.99997
        # 高斯函数邻域更新 对应了σ=σ0*e^{-t/t0}, σ0=n//10 t0=3332.83
        n = n * 0.9997

        # Check for plotting interval
        if not i % 200:
            plot_network(
                targets,
                network,
                name=data_path + '{:05d}.png'.format(i),
                axes=axes,
                obstacle=obs,
                obs_size=obs_size,
                span=span,
                fbzs=fbzs,
            )
            update_figure(axes, clean=True)

        # Check if any parameter has completely decayed. 收敛判断
        if n < 1:
            finish_info = 'Radius has completely decayed.'
            break
        if learning_rate < 0.001:
            finish_info = 'Learning rate has completely decayed.'
            break

        delta = network - old_network if old_network is not None else network
        max_delta = np.linalg.norm(delta, axis=1).max()  # 计算变化的模长 (n,1) array
        old_delta.append(max_delta)
        old_network = network.copy()
        if len(old_delta) > network.shape[0]:  # 存储神经元结点数目的delta,避免概率影响收敛
            old_delta.pop(0)
        if max(old_delta) < gate:
            # 当迭代变化最大值还小于设定的精度时就停止
            finish_info = "Average movement has reduced to {},".format(
                np.mean(old_delta) * span)
            finish_info += "max movement {},".format(np.max(old_delta) * span)
            break

    # 训练完成后进行的工作
    finish_info += "finishing execution at {} iterations".format(i)
    logging.info(finish_info)

    # 保存路径图片
    plot_network(
        targets,
        network,
        name=data_path + 'final.png',
        obstacle=obs,
        obs_size=obs_size,
        span=span,
        fbzs=fbzs,
    )

    # 计算路径距离
    distance = route_distance(network) * span  # 恢复到原坐标系下的距离
    logging.info('Route found of length %s', distance)

    return distance


class Network():
    def __init__(self, network, num, targets, radius):
        self.network = network
        self.num = num  # 神经元结点数目
        self.targets = targets
        self.radius = radius  # 用在邻域函数里面的邻域半径
        self.old_net = 0

    def get_delta(self):
        result = np.linalg.norm(self.network - self.old_net, axis=1).max()
        self.old_net = self.network.copy()
        return result


def multi_som(target,
              iterations,
              learning_rate=0.8,
              obstacle=None,
              fbzs=None,
              data_path="assets/"):
    # 读取数据并进行归一化处理
    logging.info('multi_som loading data')
    targets = target.copy()[['x', 'y']]
    obs = obstacle.copy()[['x', 'y']] if obstacle is not None else None

    norm_ans = normalization(fbzs, targets, obs)
    targets, obs, span, fbzs = norm_ans["result"][0], norm_ans["result"][
        1], norm_ans["dif"], norm_ans["fbzs"]
    # 将数据统一转化为ndarray类型
    obs = obs[['x', 'y']].to_numpy()
    targets = targets[['x', 'y']].to_numpy()

    # 一些便于程序运行的设定
    axes = update_figure()  # set up a figure
    old_delta = []
    gate = 1 / span  # 收敛条件设定，精度的映射
    obs_size = 4 * gate
    net_size = 15
    # 聚类划分环
    k = 2
    labels = cluster(targets, n=k, fbzs=fbzs)
    Network_group = []  # 按照聚类结果创建的Network
    for i in range(k):
        sub_targets = targets[labels == i]
        num = sub_targets.shape[0] * net_size
        radius = num
        sub_network = generate_network(num)
        Network_group.append(Network(sub_network, num, sub_targets, radius))

    logging.info('%s network created', len(Network_group))
    logging.info('Starting iterations:')

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")

        for net in Network_group:
            # 常规SOM
            target = random.choice(net.targets)
            winner_idx = select_closest(net.network, target)
            gaussian = get_neighborhood(winner_idx, net.radius // 10, net.num)
            target_delta = gaussian[:, np.newaxis] * (target - net.network)
            net.network += learning_rate * target_delta

            # 选取最接近目标点的获胜结点,其他结点往距离最小处移动
            winner_indices = np.apply_along_axis(
                func1d=lambda t: select_closest(net.network, t),
                axis=1,
                arr=net.targets,
            )  # 胜者不改变
            net.network = sep_and_close_nodes(
                net.network,
                decay=learning_rate,
                targets=net.targets,
                obstacle=obs,  # 圆形障碍物
                obs_size=obs_size,  # 障碍物半径
                fbzs=fbzs,  # 不规则障碍物
                gate=gate,  # 最大更新步长
                winner_indices=winner_indices,
            )

        # Decay the variables
        learning_rate = learning_rate * 0.99997
        for net in Network_group:
            net.radius *= 0.9997

        # Check for plotting interval
        if not i % 200:
            plot_network(
                targets,
                neurons=None,
                name=data_path + '{:05d}.png'.format(i),
                axes=axes,
                obstacle=obs,
                obs_size=obs_size,
                span=span,
                fbzs=fbzs,
                Networks=Network_group,
            )
            update_figure(axes, clean=True)

        # Check if any parameter has completely decayed. 收敛判断
        if max([net.radius for net in Network_group]) < 1:
            finish_info = 'Radius has completely decayed.'
            break
        if learning_rate < 0.001:
            finish_info = 'Learning rate has completely decayed.'
            break
        for net in Network_group:
            old_delta.append(net.get_delta())
            if len(old_delta) > net_size * targets.shape[0]:  # 避免概率影响收敛
                old_delta.pop(0)
        if max(old_delta) < gate:
            # 当迭代变化最大值还小于设定的精度时就停止
            finish_info = "Max movement has reduced to {},".format(
                max(old_delta) * span)
            break

    # 训练完成后进行的工作
    finish_info += "finishing execution at {} iterations".format(i)
    logging.info(finish_info)

    # 保存路径图片
    plot_network(
        targets,
        neurons=None,
        name=data_path + 'final.png',
        obstacle=obs,
        obs_size=obs_size,
        span=span,
        fbzs=fbzs,
        Networks=Network_group,
    )

    # 计算路径距离
    distance = 0
    for net in Network_group:
        distance += route_distance(net.network) * span  # 恢复到原坐标系下的距离
    logging.info('Route found of length %s', distance)

    return distance


if __name__ == '__main__':
    main()
