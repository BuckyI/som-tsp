import argparse
import os
import numpy as np

from io_helper import read_tsp, normalize, read_obs, normalization, get_gif, save_info
from neuron import generate_network, get_neighborhood, get_route, get_ob_influence, get_route_vector, ver_vec
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
    return parser.parse_args()


def main():
    # 加载 problem
    arg = get_parser()
    target = read_tsp(arg.target)  # 读取城市坐标数据
    obstacle = read_obs(arg.obstacle) if arg.obstacle is not None else None
    print("Problem loading completed.")

    # 设定本次数据存储路径
    time_id = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())  # 标识本次运行
    data_path = "assets/" + time_id + "/"  # 作为运行数据存储的路径
    os.mkdir(data_path)  # 建立文件夹

    start_time = time.process_time()

    # 获得路径结果
    distance = som(target, 100000, 0.8, obstacle,
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

    norm_ans = normalization(cities, obs)
    cities, obs, diff = norm_ans[0][0], norm_ans[0][1], norm_ans[1]
    # obs_n = obs[['x', 'y']].to_numpy()

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8  # 这里是神经元数目，别误解为人口(population)数目
    n = n + obs.shape[0] * 5 if obstacle is not None else n

    # parameters set to observe and evaluate 自己加的
    axes = update_figure()
    gate = 1 / diff  # 收敛条件设定，精度的映射
    # Generate an adequate network of neurons:
    network = generate_network(n)  # 2列矩阵
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            # "\r"回车，将光标移到本行开头，大概就是覆盖了吧
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")

        # Choose a random city
        # DataFrame.values --> numpy.ndarray
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        gaussian = get_neighborhood(winner_idx, n // 10, network.shape[0])
        city_delta = gaussian[:, np.newaxis] * (city - network)

        # choose a random obstacle
        if obs is None:
            obs_delta = 0
        else:
            obs_sample = obs.sample(1)[['x', 'y']].values
            loser_idx = select_closest(network, obs_sample)
            gaussian = get_neighborhood(loser_idx, n // 10, network.shape[0])
            obs_influence = ver_vec(
                get_route_vector(network, t=1),
                get_ob_influence(obs_sample, network, sigma=4 * gate))
            obs_delta = gaussian[:, np.newaxis] * obs_influence

        # Update the network's weights (closer to the city)
        delta = city_delta + obs_delta
        network += learning_rate * delta
        # Decay the variables
        # 学习率更新 对应了 e^{-t/t0} t0=33332.83
        learning_rate = learning_rate * 0.99997
        # 高斯函数邻域更新 对应了σ=σ0*e^{-t/t0}, σ0=n//10 t0=3332.83
        n = n * 0.9997

        # Check for plotting interval
        if not i % 200:
            plot_network(cities,
                         network,
                         name=data_path + '{:05d}.png'.format(i),
                         axes=axes,
                         obstacle=obs)
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
        delta = np.linalg.norm(delta, axis=1)  # 计算变化的模长 (n,1) array
        if delta.max() < gate:
            # 当迭代变化最大值还小于设定的精度时就停止
            print(
                "Average movement has reduced to {},".format(delta.mean() *
                                                             diff),
                "max movement {},".format(delta.max() * diff),
                "finishing execution at {} iterations".format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    # 训练完成后进行的工作
    # 保存路径图片
    plot_network(cities, network, name=data_path + 'final.png', obstacle=obs)

    # 计算路径距离
    distance = route_distance(network) * diff  # 恢复到原坐标系下的距离
    print('Route found of length {}'.format(distance))

    return distance


if __name__ == '__main__':
    main()
