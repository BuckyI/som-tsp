import pandas as pd
import numpy as np
import imageio
import os
import logging


def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame\n

    The .tsp files can be found in the TSPLIB project. Currently, the library\n
    only considers the possibility of a 2D map.\n
    """
    with open(filename) as f:
        node_coord_start = None
        dimension = None
        lines = f.readlines()  # 列表每个元素是一行

        # Obtain the information about the .tsp
        i = 0
        while not dimension or not node_coord_start:  # 循环直到获取两个值
            line = lines[i]
            if line.startswith('DIMENSION :'):
                dimension = int(line.split()[-1])  # 城市数目
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i  # 结点坐标开始处
            i = i + 1

        logging.info('Problem with %s cities read.', dimension)

        f.seek(0)  # 指针回到开头，刚才readlines到达EOF了

        # Read a data frame out of the file descriptor
        cities = pd.read_csv(
            f,
            skiprows=node_coord_start + 1,  # 跳过刚开始的几行
            sep=' ',
            names=['city', 'y', 'x'],
            dtype={
                'city': str,
                'x': np.float64,
                'y': np.float64
            },
            header=None,  # if col names are passed explicitly, header=None.
            nrows=dimension)  # Number of rows of file to read.

        # cities.set_index('city', inplace=True)  # 把 city 列设为索引，直接对源array进行修改。

        return cities


def read_obs(filename):
    """
    Read a file in .obs format into a pandas DataFrame\n
    """
    with open(filename) as f:
        start = None
        dimension = None
        lines = f.readlines()  # 列表每个元素是一行

        # Obtain the information about the .obs
        i = 0
        while not dimension or not start:  # 循环直到获取两个值
            line = lines[i]
            if line.startswith('DIMENSION :'):
                dimension = int(line.split()[-1])  # 障碍物数目
            if line.startswith('OBSTACLE_SECTION'):
                start = i  # 结点坐标开始处
            i = i + 1

        logging.info('Environment with %s obstacles read.', dimension)

        f.seek(0)  # 指针回到开头，刚才readlines到达EOF了

        # Read a data frame out of the file descriptor
        obstacles = pd.read_csv(
            f,
            skiprows=start + 1,  # 跳过刚开始的几行
            sep=' ',
            names=['obstacle', 'y', 'x'],
            dtype={
                'obstacle': str,
                'x': np.float64,
                'y': np.float64
            },
            header=None,  # if col names are passed explicitly, header=None.
            nrows=dimension)  # Number of rows of file to read.

        return obstacles


def read_fbz(filename):
    """
    filename: 文件路径名称
    return: fbzs 列表,元素为ndarray,代表一个多边形区域
    """
    with open(filename, "r") as f:
        fbzs = []
        for line in f:
            fbz = line.split(";")
            for i, v in enumerate(fbz):
                fbz[i] = [float(k) for k in v.split(",")]
            else:
                fbzs.append(np.array(fbz))

    logging.info('Environment with %s forbidden zones read.', len(fbzs))
    return fbzs


def get_gif(folder, name="result"):
    """将folder下的所有.png图片合成为一个gif文件"""
    gif_images = []
    img_paths = [folder + i for i in os.listdir(folder) if i.endswith(".png")]
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    gif_name = folder + "/" + name + ".gif"
    imageio.mimwrite(gif_name, gif_images, fps=1)


def save_info(path, **info):
    with open(path + "README.md", "w") as f:
        f.write("# README\n\n")
        # f.write("[path](./tour.tour)\n\n")
        f.write("![final](./final.png)\n\n")
        f.write("![result](./result.gif)\n\n")
        for i in info:
            f.write("---\n`{}`\n".format(str(i)))
            f.write(str(info[i]) + "\n\n")


def normalize(points):
    """
    points: DataFrame type `normalize(cities[['x', 'y']])`\n
    return: DataFrame type\n

    Return the normalized version of a given vector of points.\n

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.\n
    """
    # 这部分其实挺奇怪的，为啥不直接用 max(points.max()-points.min()) 呢
    # get a numpy.ndarray:
    # if DeltaX > DeltaY, then ratio=[1, DeltaY/DeltaX]
    # if DeltaX < DeltaY, then ratio=[DeltaX/DeltaY, 1]
    ratio = (points.x.max() - points.x.min()) / (points.y.max() -
                                                 points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)

    # Apply a function along an axis(default 0) of the DataFrame.
    # get a DataFrame (x/deltaX, y/deltaY) delta=max-min
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    # p: each series ['x','y'] ; result: xy等比例缩放 max{DeltaX, DeltaY}
    return norm.apply(lambda p: ratio * p, axis=1)


def normalization(fbzs=None, *dfs):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!待完善
    *dfs: DataFrame list
    return: ([*fixed_dfs],max_dif)
    max_dif: 所有坐标中最大的跨度,根据这个对整体缩小
    """
    diff = []
    offset_x = []
    offset_y = []
    for i in dfs:
        if i is None:
            continue
        diff.append(max(i.max() - i.min()))
        offset_x.append(i.min()[0])
        offset_y.append(i.min()[1])

    for i in fbzs:
        diff.append(max(i.max(axis=0) - i.min(axis=0)))  # x,y最大跨度
        offset_x.append(i.min(axis=0)[0])
        offset_y.append(i.min(axis=0)[1])

    dif = max(diff)
    offset = (min(offset_x), min(offset_y))
    result = []
    for i in dfs:
        if i is None:
            result.append(None)
            continue
        temp = (i[['x', 'y']] - offset) / dif
        result.append(temp)

    for i, v in enumerate(fbzs):
        fbzs[i] = (v - offset) / dif

    info = {"result": result, "dif": dif, "fbzs": fbzs}
    return info
