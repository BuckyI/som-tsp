import numpy as np


def select_closest(candidates, origin):
    """
    candidates: [numpy.ndarray] size*2 it is actually all the neurons\n
    origin: [numpy.ndarray] 1*2 it is the chosen city\n
    return: [int] at the given city, the nearest neuron's index\n
    
    检索array中最小值的位置，并返回其下标值，即找到最相似neuron\n
    Return the index of the closest candidate to a given point.\n
    """
    return euclidean_distance(candidates, origin).argmin()


def euclidean_distance(a, b):
    """
    a, b = candidates, origin\n
    求a-b每一行的算数平方和开根号，也就是行向量之间的欧几里德距离了。\n
    Return the array of distances of two numpy arrays of points.\n
    """
    return np.linalg.norm(a - b, axis=1)


def route_distance(cities, index=None):
    """
    cities: [DataFrame] 排好顺序的城市\n
    index: [index] 如果给了这个参数,就根据index排序之后再计算距离
    Return the cost of traversing a route of cities in a certain order.\n
    """
    if index is not None:
        cities = cities.reindex(index)
        
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    # 按行移1位后比较距离，也就是相邻 city 的距离了
    return np.sum(distances)
