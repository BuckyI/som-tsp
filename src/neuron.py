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


def get_ob_influence(ob, point, sigma=10, k=0.434):
    """
    k*sigma^2 determines the range of gaussian
    """
    distances = ob - point
    weighed_influence = np.exp(-distances**2 / (k * sigma**2))
    return weighed_influence.sum(axis=0)
