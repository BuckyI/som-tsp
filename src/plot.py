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
        fig = plt.figure()
        # axis = fig.add_axes([0, 0, 1, 1])  # axes 与 figure 相同大小，完全覆盖
        axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal', adjustable='datalim')  # equal: 正方形
        # axis.axis('off')

    axis.legend()
    axis.figure.canvas.flush_events()
    if clean:  # 窗口方式运行时，只有flush之后才会更新，因此此时窗口不会被清空
        axis.cla()
        axis.set_aspect('equal', adjustable='datalim')  # equal: 正方形
        # axis.axis('off')

    return axis


def plot_network(cities, neurons, name='diagram.png', ax=None):
    """
    cities: [DataFrame] 归一化之后的
    neurons: [ndarray] the network
    name: [str] filepath
    ax: [axes] unused 如果传入一个 axes Object，将在其上作图并返回，而不是保存到文件
    
    Plot a graphical representation of the problem
    """
    mpl.rcParams['agg.path.chunksize'] = 10000  # 增大数据块大小，略微加快速度，避免渲染失败

    if not ax:
        plt.ioff()
        # settings
        fig = plt.figure(figsize=(5, 5), frameon=False)  # 5X5 inch 没有边框 dpi=80
        axis = fig.add_axes([0, 0, 1, 1])  # axes 与 figure 相同大小，完全覆盖

        axis.set_aspect('equal', adjustable='datalim')  # equal: 正方形
        plt.axis('off')  # 关闭坐标轴 建议改成：axis.axis('off')

        # city 以点的形式
        axis.scatter(cities['x'], cities['y'], color='red', s=4, label="city")
        # neuron 以线的形式
        axis.plot(
            neurons[:, 0],  # x
            neurons[:, 1],  # y
            'r.',  # red dot (actually it's blue!)
            ls='-',  # line style -
            color='#0063ba',
            label="neuron",
            markersize=2)  # the s of marker is 4
        axis.legend()
        # save 边框  tight 内边距 0
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        plt.ion()
    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4, label="city")
        ax.plot(neurons[:, 0],
                neurons[:, 1],
                'r.',
                ls='-',
                color='#0063ba',
                label="neuron",
                markersize=2)
        return ax


def plot_route(cities, route, name='diagram.png', ax=None):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        plt.ioff()
        # settings
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')
        # 画出城市点
        axis.scatter(cities['x'], cities['y'], color='red', s=4, label="city")
        # old route is pd's index Object, new route is ordered cities
        route = cities.reindex(route)
        # 在 DataFrame route 末尾添加一行，赋开头的值，这样路线首尾相连
        # 为什么画 network 不需要呢？是因为 neurons 太多了看不出来首尾没连~
        route.loc[route.shape[0]] = route.iloc[0]
        # 画出路径线
        axis.plot(route['x'],
                  route['y'],
                  color='purple',
                  linewidth=1,
                  label="route")
        axis.legend()
        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        plt.ion()
    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4, label="city")
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax
