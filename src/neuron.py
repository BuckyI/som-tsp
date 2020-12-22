import numpy as np
import logging
from distance import select_closest


def test_plot(arr1, arr2=None, filename="test"):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.axis('equal')
    plt.scatter(arr1[:, 0], arr1[:, 1], color='#4169E1', s=0.1)
    if arr2 is not None:
        plt.scatter(arr2[:, 0], arr2[:, 1], color='#DC143C', s=0.1)
        plt.quiver(
            arr1[:, 0],  # X
            arr1[:, 1],  # Y
            (arr2 - arr1)[:, 0],  # U
            (arr2 - arr1)[:, 1],  # V
            angles='xy',
            scale_units='xy',
            scale=1,  # 长短
            units='xy',
            width=0.003,  # 粗细
            pivot='tail',
            color="#3CB371")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=400)


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


def unit_ver_vec(vector):
    """get unit vertical vector 对于零向量仍然返回零向量"""
    v = vector.copy()
    v[:, 0], v[:, 1] = v[:, 1], -v[:, 0]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[np.isnan(v)] = 0
    return v


def unit_vector(vector):
    "vector.shape==(n,2) 返回单位向量"
    v = vector.copy()
    a = np.linalg.norm(v, axis=1, keepdims=True)
    indices = a[:, 0] != 0  # 找到不是零向量的
    v[indices] = v[indices] / a[indices]
    return v


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


def get_troubled_indices(network, div=1, step=0, v=None, **environment):
    """
    div int; v ndarray
    用来确定network结点连线分成div份,每个等分点也都没在障碍物里
    v是从network node_i 出发到 node_i+1 的delta
    step:沿着unit_v最多走step=div-1步,因为第div步就是下一个结点了
    step初始为0,表示源节点
    """
    if div == 1 or v is None:  # 表示不对节点之间线段上的点进行判断
        return np.apply_along_axis(is_node_in_trouble, 1, network,
                                   **environment)
    elif step == div - 1:  # 到达了需要判断的最后一个点
        return np.apply_along_axis(is_node_in_trouble, 1, network,
                                   **environment)
    else:
        # 获得处于障碍物内部的结点的索引
        indices = np.apply_along_axis(is_node_in_trouble, 1,
                                      network + step * v / div, **environment)

        new_indices = indices == False  # 判断为不在障碍物内的点
        if new_indices.any():
            # 如果还存在判断为不在障碍物内的点,进行一次更加严苛的迭代判断
            indices[new_indices] = get_troubled_indices(
                network=network[new_indices],
                div=div,
                step=step + 1,  # 走的更远1步,一直到div-1步,如果都没在障碍物内,才算不在
                v=v[new_indices],
                **environment,
            )
        return indices


def get_away(network, step, head_dir, k=0, max_k=5, **environment):
    "network 沿着 dir 方向脱离障碍物, 最大步长为 max_k*step"
    # 调用 get_troubled_indices 的参数
    div = 2
    v = get_route_vector(network)
    # step 1 update and find the bad nodes
    if k == 0:
        # k==0 时更新数值为0,省去第一轮计算
        new_index = get_troubled_indices(network, div=div, v=v, **environment)
    elif k > max_k:
        # k 过大时,不进行更新,限制get_away的更新幅度(最大步长)
        new_index = np.array(False)
    else:
        # 更新
        up = network + k * step * head_dir
        down = network - k * step * head_dir
        good_up = ~get_troubled_indices(up, div=div, v=v, **environment)
        good_down = ~get_troubled_indices(down, div=div, v=v, **environment)
        network[good_down] = down[good_down]
        network[good_up] = up[good_up]

        new_index = ~(good_down | good_up)  # 仍然位于障碍物内的点

    # step 2 update the bad nodes
    if new_index.any():  # 存在处于障碍物内的点
        network[new_index] = get_away(
            network=network[new_index],
            step=step[new_index],
            head_dir=head_dir[new_index],
            k=k + 1,
            max_k=max_k,
            **environment,
        )

    return network


def sep_and_close_nodes(network, r=1, decay=1, **environment):
    """
    r 是邻域半径
    s-->m-->d
    m:network
    vx表示垂直向量,从sd中点指向x
    """
    # def get_direction(target, s, sd, vm, max_distance):
    #     """
    #     这是一个和主函数高度耦合的函数 return (n,2)
    #     target: (2,)
    #     s, sd, vm: are the information of network
    #     max_distance: (n,) determines how far the node would go
    #     """
    #     st = target - s
    #     vt = ver_vec(sd, st)

    #     indices1 = is_target_between_line(sd, st)  # target在线段sd中间的
    #     indices2 = np.linalg.norm(vt - vm,
    #                               axis=1) < max_distance  # target距离m不远的
    #     vt[~(indices1 & indices2)] = np.inf  # 不在线段中间或者距离太远的,就设为inf不影响
    #     return vt

    gate = environment.get("gate", 1)  # 最大单次步长

    m = network
    s = np.roll(m, 1, axis=0)
    d = np.roll(m, -1, axis=0)
    sd = d - s  # 衡量区域宽度,对区域进行分割的依据
    step = (np.linalg.norm(sd, axis=1, keepdims=True) * 0.1).clip(
        0.01 * gate, gate) * decay  # 一步的步长
    max_distance = 5 * r * step  # 上下最远范围

    vm = ver_vec(sd, m - s)  # sm 垂直 sd 分解
    # targets = environment.get("targets", None)
    # if targets is not None:
    #     vts = np.apply_along_axis(
    #         func1d=lambda t: get_direction(t, s, sd, vm, 0.1 *
    #                                        max_distance[:, 0]),
    #         axis=1,
    #         arr=targets,
    #     )
    #     indices = np.argmin(np.linalg.norm(vts, axis=2), axis=0)  # 模最小值所在索引
    #     vt = vts[indices, np.arange(indices.shape[0]), :]
    #     vt[vt == np.inf] = 0  # 无穷大的目标点t意味着不可达,置零
    # else:
    #     indices = np.array(False)
    #     vt = [0, 0]
    vt = [0, 0]
    head_dir = vt - vm
    unit_head_dir = unit_vector(head_dir)  # 前进的方向,决定node上下移动单位距离

    # 水平分散
    # decay越接近0,收敛越快,越容易震荡,但是后期形状基本定型,避免了震荡
    base_net = s + 0.5 * sd + decay * vm
    # base_net = s + 0.5 * sd + vm
    # base_net = s + 0.5 * sd  # 这种方法舍弃了结点之前的经验,收敛速度更快,但是有震荡的风险

    # node 往看到的目标点走一步
    # base_net[indices] = base_net[indices] + unit_head_dir * step[indices]
    # 避障
    result = get_away(
        base_net,
        step,
        unit_head_dir,
        k=0,
        # max_k=5 * r / decay,  # 后面单个步长短但是最大总步长变大
        max_k=np.inf,
        **environment,
    )
    # 虽然是从0开始,但是线会自然变直

    winner_indices = environment.get("winner_indices", None)  # 胜者不动
    if winner_indices is not None:
        result[winner_indices] = network[winner_indices]

    return result


def is_target_between_line(sd, st):
    "判断t是否在sd的两点之间的带状空间,不包含边界"
    # 注释掉的是之前的方法,需要三个参数
    # inside = (sd * st).sum(axis=1) * (sd * dt).sum(axis=1) < 0
    x = (sd * st).sum(axis=1)  # st 投射在 sd 上的长度

    return (x > 0) & (x < np.linalg.norm(sd, axis=1))


def is_node_in_trouble(node, **environment):
    """"""
    fbzs = environment.get("fbzs", None)
    if fbzs is not None:
        for fbz in fbzs:
            if is_point_in_polygon(node, fbz) == 1:
                return True

    obstacle = environment.get("obstacle", None)
    obs_size = environment.get("obs_size", 1)
    if obstacle is not None:
        distances = np.linalg.norm(node - obstacle, axis=1)
        if distances[distances < obs_size].any():
            return True

    return False


def is_point_in_polygon(point, arr):
    """
    point: x,y=point
    arr:[ndarray] (n,2) polygon
    return:
    0: point is on the boundary;
    1: point is in the polygon;
    -1: point is not on the boundary;
    """
    max_x, max_y = arr.max(axis=0)
    min_x, min_y = arr.min(axis=0)
    if not min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
        return -1  # the point is outside the box

    point_start = arr[-1]  # from the last one to start (circle)
    x, y = point  # take the x, y coordinates
    count = 0
    for i in arr:
        point_end = i
        # 点与多边形顶点重合
        if (x == point_start[0]
                and y == point_start[1]) or (x == point_end[0]
                                             and y == point_end[1]):
            return 0
        # 点与多边形水平边界线重合
        if point_end[1] == point_start[1] and y == point_start[1]:
            return 0
        # 判断线段两端点是否在射线两侧
        if min(point_end[1], point_start[1]) <= y <= max(
                point_end[1], point_start[1]):
            # 这部分无法检测出在水平边界线上的点,所以之前排除掉了
            # 线段上与射线的 Y 坐标相同的点的 X 坐标
            x0 = point_end[0] - (point_end[1] - y) * (
                point_end[0] - point_start[0]) / (point_end[1] -
                                                  point_start[1])
            # 点在多边形的边上
            if x0 == x:
                return 0
            elif x0 > x:  # 水平向右射线穿过多边形的边界1次
                count += 1

        point_start = point_end
    else:
        return -1 if (count % 2) == 0 else 1
