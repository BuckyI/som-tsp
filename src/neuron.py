import numpy as np

from distance import select_closest


def generate_network(size):
    """
    size: [int] the number of neurons / the length of array.\n
    return: [numpy.ndarray] size*2  0~1 \n

    Generate a neuron network of a given size.\n

    Return a vector of two dimensional points in the interval [0,1]. \n
    """
    return np.random.rand(size, 2)


def get_neighborhood(center, radix, domain):
    """
    center: [int] winner_idx 是被选中的神经元的索引位置\n
    radix: [int] 基数，邻域半径 original: number of neurons // 10 , 随迭代缩减 \n
    domain: [int] network.shape[0], 8 times of cities 领域\n
    return: [numpy.ndarray] the neighborhood function value vector [0,1]\n
    基本上就是每个位置神经元的邻域函数更新幅度。

    Get the range gaussian of given radix around a center index.\n
    """

    # Impose an upper bound on the radix to prevent NaN and blocks
    # radix 每次循环会缩小，这一部分在传入的 radix 为 0 时进行修正
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    # np.arange(domain) 产生从0到domain间隔为1的等差向量，也就是各神经元本来的位置索引
    # delta=被选中神经元与本来索引的差的绝对值，即距离
    # domain-delta=获得另一个距离（感觉是通过找关系得知可以这样算的）
    # 两个距离取最小，得到的是环形的拓扑结构下的距离。
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances * distances) / (2 * (radix * radix)))


def get_route(cities, network):
    """
    cities: [DataFrame] the normalized set of cities ['city', 'y', 'x']\n
    network: [numpy.ndarray] the trained network with 8*cities neuron\n
    return: [Index] the index of cities\n
    如果有多个城市被分配到了同一个神经元，那么按照下面的代码排序的时候，这几个会按照之前的顺序（所以这里不是最优）\n
    Return the route computed by a network.\n
    """
    cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest(network, c),  # 计算结果存在新 col winner 里
        axis=1,  # 按行 index 迭代
        raw=True  # the passed function will receive ndarray objects.
    )
    # 先按照'winner'排序（神经元顺序），然后返回index，即对应的城市顺序
    return cities.sort_values('winner').index


def get_ob_influence(obs, node, sigma=10):
    """
    k*sigma^2 determines the range of gaussian
    sigma: 可以看做一个邻域范围
    note: obs是(n,2),node是(2,) 返回所有obs对node的影响
    如果删去sum,可以计算一个 obs (2,)对所有 node (n,2) 的影响
    """
    difference = node - obs
    distances = np.linalg.norm(difference, axis=1)
    # 沿方向向量移动的距离,为负时置零
    fix_dist = (sigma - distances).clip(0)
    # 获得单位方向向量
    distances[distances > sigma] = np.inf  # 超出sigma范围以外的设为无穷大不处理
    vec = difference / distances[:, np.newaxis]
    # 计算影响,这里障碍物视为圆形
    influence = vec * fix_dist[:, np.newaxis]

    # 不使用高斯函数了
    # influence = -np.exp(
    #     -distances**2 / (2 * sigma**2)
    # )[:, np.newaxis] * difference / distances[:, np.newaxis] * sigma
    return influence.sum(axis=0)  # 所有的影响向量合并


def get_ob_influences(network, obstacle, radius):
    """
    radius: 障碍物视为圆形,radius是半径
    """

    influences = np.apply_along_axis(
        func1d=lambda p: get_ob_influence(obstacle, p, radius),  # 每个点独自面对所有障碍物
        axis=1,
        arr=network)
    return influences


def ver_vec(direction, vector):
    """
    向量垂直分解
    direction: [ndarray] 方向向量
    vector: [ndarray] 要分解的向量
    return: vector 垂直于 direction 的分向量.
    两个array相同行,对应行求垂直向量.
    """
    x0, y0 = direction[:, 0], direction[:, 1]
    x1, y1 = vector[:, 0], vector[:, 1]
    x = y0**2 * x1 - x0 * y0 * y1
    y = x0**2 * y1 - x0 * y0 * x1
    return np.array([x, y]).T / (x0**2 + y0**2)[:, np.newaxis]


def get_route_vector(network, d=0, t=0):
    """
    network: [ndarray] 坐标点矩阵
    d: direction 0顺时针 1逆时针
    t==0: vector[i,:]是从network[i,:]出发的位移向量
    t==1: vector[i,:]是到达network[i,:]的位移向量
    return vector [ndarray]
    根据坐标点矩阵获得依次相连的线段/向量矩阵
    """
    if d == 0:  # 顺时针
        vector = network - np.roll(network, 1, axis=0)
        if t == 0:
            vector = np.roll(vector, -1, axis=0)
    elif d == 1:  # 逆时针
        vector = np.roll(network, 1, axis=0) - network
        if t == 1:
            vector = np.roll(vector, -1, axis=0)
    return vector


def sepaprate_node(network):
    """
    return a seperated network with (more) even interval
    按顺时针方向进行操作
    """
    # 为了便于理解,使用字母 abc 表示
    # ab = get_route_vector(network, 0, 0)
    ab = np.roll(network, -1, axis=0) - network
    ac = np.roll(network, -2, axis=0) - network
    ab_new = ver_vec(ac, ab) + 0.5 * ac
    delta = ab_new - ab  # 施加于b的更新向量
    return network + np.roll(delta, 1, axis=0)
