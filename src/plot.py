import matplotlib.pyplot as plt
import matplotlib as mpl


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
        fig = plt.figure(figsize=(5, 5), frameon=False)
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
        fig = plt.figure(figsize=(5, 5), frameon=False)  # 5X5 inch 没有边框 dpi=80
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
    # city 以点的形式
    axes.scatter(cities['x'], cities['y'], color='red', s=4, label="city")
    # path 以线的形式
    try:
        x, y = path["x"], path["y"]
    except Exception:
        x, y = path[:, 0], path[:, 1]
    axes.plot(
        x,  # x
        y,  # y
        'r.',  # red dot (actually it's blue!)
        ls='-',  # line style -
        color='#0063ba',
        label="path",
        linewidth=1,
        markersize=2)  # the s of marker is 4
    if environment.get("obstacle", None) is not None:
        obs = environment["obstacle"]
        axes.scatter(obs['x'], obs['y'], color='y', s=4, label="obstacle")
    # 更新标签
    axes.legend()
    return axes


def plot_route(cities, route, name='diagram.png', ax=None, **environment):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        plt.ioff()
        # settings
        fig = plt.figure(figsize=(5, 5), frameon=False)
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
