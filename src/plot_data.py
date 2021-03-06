import matplotlib.pyplot as plt
import matplotlib as mpl
from neuron import get_route_vector
import numpy as np


def test_plot(arr1, arr2, filename="test"):
    import matplotlib.pyplot as plt
    plt.axis('equal')
    plt.scatter(arr1[:, 0], arr1[:, 1], color='r', s=0.1)
    plt.scatter(arr2[:, 0], arr2[:, 1], color='y', s=0.1)
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
        color="#6495ED")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)


def input_points(x_range=(0, 100),
                 y_range=(0, 100),
                 title="left click: get point, right: cancel, mid: finish"):
    """
    x_range, y_range: [tuple] (start,end)
    title: [str] information of the points
    return: list of tuple (x, y)
    """
    fig = plt.figure()
    axis = fig.subplots()
    axis.set_xlim(x_range[0], x_range[1])
    axis.set_ylim(y_range[0], y_range[1])
    axis.set_title(title)
    result = axis.figure.ginput(-1)
    plt.close()
    return result


def update_figure(axis=None, clean=False):
    """
    axis: [matplotlib.axes._axes.Axes] The axis to be updated
    clean: [bool] If True, the old axes will be cleaned
    return: axis
    plz change your figure at elsewhere, and use this to update the screen.
    update a current figure(axis) in interactive mode
    if not given, create one and return it
    """
    if not plt.isinteractive():  # 打开交互模式
        plt.ion()
    if not axis:
        fig = plt.figure(figsize=(4.5, 4.5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])  # axes 与 figure 相同大小，完全覆盖
        # axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal', adjustable='datalim')  # equal: 正方形
        axis.axis('off')

    # axis.legend()
    axis.figure.canvas.flush_events()
    if clean:  # 窗口方式运行时，只有flush之后才会更新，因此此时窗口不会被清空
        axis.cla()
        axis.set_aspect('equal', adjustable='datalim')  # equal: 正方形
        axis.axis('off')

    return axis


def plot_network(cities, neurons, name='diagram.png', axes=None,
                 **environment):
    """
    cities: [DataFrame] 归一化之后的
    neurons: [ndarray] the network
    name: [str] filepath
    axes: [axes] unused 如果传入一个 axes Object，将在其上作图并返回，而不是保存到文件
    
    Plot a graphical representation of the problem
    """
    mpl.rcParams['agg.path.chunksize'] = 10000  # 增大数据块大小，略微加快速度，避免渲染失败

    if not axes:
        plt.ioff()
        # settings
        fig = plt.figure(figsize=(4.5, 4.5),
                         frameon=False)  # 5X5 inch 没有边框 dpi=80
        axes = fig.add_axes([0, 0, 1, 1])  # axes 与 figure 相同大小，完全覆盖

        axes.set_aspect('equal', adjustable='datalim')  # equal: 正方形
        plt.axis('off')  # 关闭坐标轴 建议改成：axis.axis('off')

        plot_process(axes, cities, neurons, environment)
        # save 边框  tight 内边距 0
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        plt.ion()
    else:
        plot_process(axes, cities, neurons, environment)
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)  # 测试用
        return axes


def plot_process(axes, cities, path, environment={}):
    """
    cities: [DataFrame] 归一化之后的
    path: [ndarray]/[DataFrame] the network or the result route
    axes: [axes] unused 如果传入一个 axes Object，将在其上作图并返回，而不是保存到文件
    environment: [dict] of DataFrame/ndarray
    """
    def draw_network(path, axes):
        # path 以有向线段的形式
        if type(path).__name__ == 'DataFrame':
            path = path[['x', 'y']].to_numpy()
        vec = get_route_vector(path, d=0, t=0)
        axes.quiver(
            path[:, 0],  # X
            path[:, 1],  # Y
            vec[:, 0],  # U
            vec[:, 1],  # V
            angles='xy',
            scale_units='xy',
            scale=1,  # 长短
            units='xy',
            width=0.003,  # 粗细
            pivot='tail',
            color="#6495ED")

    span = environment.get("span", None)
    # 画一条线标识最小精度
    axes.plot([0, 0], [0, 1 / span], linewidth=4, label="accuracy")
    # city 以点的形式
    if type(cities).__name__ == 'DataFrame':
        cities = cities[['x', 'y']].to_numpy()
    axes.scatter(cities[:, 0], cities[:, 1], color='red', s=4, label="target")
    # 画network
    if path is not None:
        draw_network(path, axes)
    else:
        Network_group = environment.get("Networks", None)
        if Network_group is not None:
            for net in Network_group:
                draw_network(net.network, axes)

    obs = environment.get("obstacle", None)
    obs_size = environment.get("obs_size", None)
    if obs is not None:
        if type(obs).__name__ == 'DataFrame':
            obs = obs[['x', 'y']].to_numpy()
        for i in obs:
            circle = plt.Circle(i,
                                obs_size,
                                color='#FFFACD',
                                fill=True,
                                alpha=0.5)
            axes.add_artist(circle)
        axes.scatter(obs[:, 0],
                     obs[:, 1],
                     color='#FFD700',
                     s=4,
                     label="obstacle")

    fbzs = environment.get("fbzs", None)
    if fbzs is not None:
        for fbz in fbzs:
            fbz = np.row_stack((fbz, fbz[0]))  # 末尾添加开头
            plt.plot(fbz[:, 0], fbz[:, 1], color="#FFB6C1", label="forbidzone")

    ## 测试用代码
    # fbz = environment.get("fbz", None)
    # delta = environment.get("delta", None)
    # if delta is not None and fbz is not None:
    #     axes.quiver(
    #         fbz[:, 0],  # X
    #         fbz[:, 1],  # Y
    #         delta[:, 0],  # U
    #         delta[:, 1],  # V
    #         angles='xy',
    #         scale_units='xy',
    #         scale=1,  # 长短
    #         units='xy',
    #         width=0.003,  # 粗细
    #         pivot='tail',
    #         color="#6495ED")
    # 更新标签
    axes.legend()
    # update_figure(axes, clean=True)
    return axes


def plot_route(cities, route, name='diagram.png', ax=None, **environment):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        plt.ioff()
        # settings
        fig = plt.figure(figsize=(4.5, 4.5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        plot_process(axis, cities, route, environment)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        plt.ion()
    else:
        plot_process(axis, cities, route, environment)
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)  # 测试用
        return ax
