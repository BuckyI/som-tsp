import numpy as np
import pandas as pd
import os


def generate_target(x_range=(0, 100), y_range=(0, 100)):
    """
    generate the travel target for the sales man
    """
    tar_num = 100
    arr = np.zeros((tar_num, 2))
    arr[:, 0] = np.linspace(x_range[0], x_range[1], tar_num)
    arr[:, 1] = np.linspace(y_range[0], y_range[1], tar_num)

    arr_pd = pd.DataFrame(arr)
    return arr_pd


def generate_tsp(data=None, filename="assets/arr.tsp"):
    """
    trans panda.DataFrame to .tsp file
    """
    if not data:
        data = generate_target()

    info = {
        "NAME": os.path.basename(filename),
        "COMMENT": "hello world",
        "DIMENSION": data.shape[0],  # 行数
    }

    with open(filename, "w") as f:
        for key in info:
            f.write(key + " : " + str(info[key]) + "\n")
        f.write("NODE_COORD_SECTION\n")
    data.to_csv(filename,
                sep=" ",
                na_rep="0",
                float_format="%.4f",
                header=False,
                mode="a+")


if __name__ == "__main__":
    generate_tsp()
