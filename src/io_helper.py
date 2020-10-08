import pandas as pd
import numpy as np


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

        print('Problem with {} cities read.'.format(dimension))

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
            nrows=dimension # Number of rows of file to read. )

        # cities.set_index('city', inplace=True)

        return cities


def normalize(points):
    """
    points: DataFrame type `normalize(cities[['x', 'y']])`\n
    return: DataFrame type\n

    Return the normalized version of a given vector of points.\n

    For a given array of n-dimensions, normalize each dimension by removing the\n
    initial offset and normalizing the points in a proportional interval: [0,1]\n
    on y, maintining the original ratio on x.\n
    """
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
