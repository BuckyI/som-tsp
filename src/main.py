import argparse
import os
import numpy as np

from io_helper import read_tsp, normalize, read_obs, normalization, get_gif, save_info, read_fbz
from neuron import generate_network, get_neighborhood, get_route, get_ob_influences, get_route_vector, ver_vec, sepaprate_node, is_point_in_polygon
from distance import select_closest, route_distance  # , euclidean_distance
from plot import plot_network, plot_route, update_figure
from gene_tsp import generate_tour
import time


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
    # 加载 problem
    arg = get_parser()
    target = read_tsp(arg.target)  # 读取城市坐标数据
    obstacle = read_obs(arg.obstacle) if arg.obstacle is not None else None
    fbzs = read_fbz(arg.forbidzone) if arg.forbidzone is not None else None
    print("Problem loading completed.")

    # 设定本次数据存储路径
    time_id = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 标识本次运行
    data_path = "assets/" + time_id + "/"  # 作为运行数据存储的路径
    os.mkdir(data_path)  # 建立文件夹

    start_time = time.process_time()

    # 获得路径结果
    distance = som(target, 100000, 0.8, obstacle, fbzs,
                   data_path)  # from neuron 0 开始的路径 index

    run_time = time.process_time() - start_time
    print('SOM training completed. Running time: %s Seconds' % (run_time))

    # 生成相关文件
    # generate_tour(route_index,
    #               filename=data_path + "tour.tour",
    #               length=distance,
    #               comment=time_id)
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

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8  # 这里是神经元数目，别误解为人口(population)数目
    n = n + obs.shape[0] * 5 if obstacle is not None else n

    # parameters set to observe and evaluate 自己加的
    axes = update_figure()
    old_delta, old_network = [], 0  # 用来判断变化大小的收敛变量
    gate = 1 / span  # 收敛条件设定，精度的映射
    obs_size = 4 * gate
    # Generate an adequate network of neurons:
    network = generate_network(n)  # 2列矩阵
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            # "\r"回车，将光标移到本行开头，大概就是覆盖了吧
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")

        route_dir_vec = get_route_vector(network, d=0, t=1)  # 当前路径下的顺时针,出发方向向量

        # Choose a random city
        # DataFrame.values --> numpy.ndarray
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])
        city_delta = gaussian[:, np.newaxis] * (city - network)
        network += learning_rate * city_delta

        # choose a random obstacle
        if obs is None:
            obs_delta = 0
        else:
            # obs_influence = ver_vec(np.roll(route_dir_vec, 1, axis=0),
            #                         get_ob_influences(network, obs, obs_size))
            obs_delta = get_ob_influences(network, obs, obs_size)
            network += learning_rate * obs_delta

        # adjust the forbidden area
        if fbzs is not None:
            fbzs_delta = np.zeros(network.shape)
            for fbz in fbzs:
                for index, node in enumerate(network):
                    if is_point_in_polygon(node, fbz) == 1:
                        ver_dist_v = ver_vec(get_route_vector(fbz), fbz - node)
                        # 计算 node to 边界的距离并找到最小值的位置
                        ver_dist = np.linalg.norm(ver_dist_v, axis=1)
                        closest = ver_dist.argmin()

                        # update delta
                        fbzs_delta[index] += ver_dist_v[closest]
            #             # 这里可以添加安全距离 / ver_dist[closest] * 1
            # a = np.linalg.norm(fbzs_delta, axis=1)
            # fbzs_delta = ver_vec(np.roll(route_dir_vec, 1, axis=0),
            #                      fbzs_delta)  # 垂直方向影响
            # b = np.linalg.norm(fbzs_delta, axis=1)
            # b[b == 0] = np.inf
            # fbzs_delta = fbzs_delta * a[:, np.newaxis] / b[:, np.newaxis]
            # network += learning_rate * fbzs_delta

        # Update the network's weights (closer to the city)
        # delta = city_delta + obs_delta
        # network += learning_rate * delta

        # 修正结点分布,使之间隔更加均匀
        network = sepaprate_node(network)

        # Decay the variables
        # 学习率更新 对应了 e^{-t/t0} t0=33332.83
        learning_rate = learning_rate * 0.99997
        # 高斯函数邻域更新 对应了σ=σ0*e^{-t/t0}, σ0=n//10 t0=3332.83
        n = n * 0.9997

        # Check for plotting interval
        if not i % 200:
            plot_network(
                cities,
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
            print('Radius has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
                  'at {} iterations'.format(i))
            break

        delta = network - old_network if old_network is not None else network
        max_delta = np.linalg.norm(delta, axis=1).max()  # 计算变化的模长 (n,1) array
        old_delta.append(max_delta)
        old_network = network.copy()
        if len(old_delta) > network.shape[0]:  # 存储神经元结点数目的delta,避免概率影响收敛
            old_delta.pop(0)
        if max(old_delta) < gate:
            # 当迭代变化最大值还小于设定的精度时就停止
            print(
                "Average movement has reduced to {},".format(
                    np.mean(old_delta) * span),
                "max movement {},".format(np.max(old_delta) * span),
                "finishing execution at {} iterations".format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    # 训练完成后进行的工作
    # 保存路径图片
    plot_network(
        cities,
        network,
        name=data_path + 'final.png',
        obstacle=obs,
        obs_size=obs_size,
        span=span,
        fbzs=fbzs,
    )

    # 计算路径距离
    distance = route_distance(network) * span  # 恢复到原坐标系下的距离
    print('Route found of length {}'.format(distance))

    return distance


if __name__ == '__main__':
    main()
