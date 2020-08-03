import numpy as np

from distance import select_closest


def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)


def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""
    # center 是被选中的神经元的索引位置
    # radix 基数，这里应该是邻域半径
    # domain 领域，实际上是输入向量维度，也就是城市数目

    # Impose an upper bound on the radix to prevent NaN and blocks
    # 向基数（radix）施加一个最大值，阻止**
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    # np.arange(domain) 产生从0到domain向量，也就是各神经元本来的位置索引
    # delta=被选中神经元与本来索引的差的绝对值，即距离
    # domain-delta=获得另一个距离（感觉是通过找关系得知可以这样算的）
    # 两个距离取最小，得到的是环形的拓扑结构下的距离。
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))


def get_route(cities, network):
    """Return the route computed by a network."""
    cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest(network, c),
        axis=1, raw=True)

    return cities.sort_values('winner').index
