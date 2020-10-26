import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import tkinter as tk


def arr_plot(arr, obs):
    "for testing useful for Dataframe and ndarray"

    def get_xy(arr):
        try:
            x = arr.values[:, 0].tolist()
            y = arr.values[:, 1].tolist()
        except AttributeError:
            x = arr[:, 0].tolist()
            y = arr[:, 1].tolist()
        return x, y

    data = get_xy(arr)
    plt.scatter(data[1], data[0], color="red")
    data = get_xy(obs)
    plt.scatter(data[1], data[0], color="black")
    plt.show()


def generate_target(x_range=(0, 100), y_range=(0, 100), way=None):
    """
    generate the travel target for the sales man
    """
    def gui_get_points():
        """generate a gui window and click to get a list of points"""
        def callback(event):
            x, y = event.x, event.y
            var.set(str(x) + "," + str(y))
            print("\rclick position", x, y, end="")
            points.append((event.y, event.x))  # 为了适应.tsp文件中的坐标顺序

        points = []

        window = tk.Tk()
        window.title('get points')
        var = tk.StringVar(value="hello")

        fr = tk.Frame(window,
                      width=200,
                      height=200,
                      highlightthickness=1,
                      bg="yellow")
        fr.bind("<Button -1>", callback)
        fr.pack(side="top")
        tk.Label(window, textvariable=var, font=('Arial', 12)).pack()
        tk.Button(window, text="Quit", command=window.destroy).pack()
        tk.mainloop()
        return points

    def get_circle():
        tar_num = 100
        arr = np.zeros((tar_num, 2))
        t = np.arange(0, tar_num, 1)
        arr[:, 0] = np.linspace(x_range[0], x_range[1], tar_num)
        arr[:, 1] = np.linspace(y_range[0], y_range[1], tar_num)

        fix_point = np.array([x_range[1], y_range[0]])
        delta = fix_point - arr
        delta = delta / np.linalg.norm(delta, axis=1)[:, np.newaxis]
        weights = -0.0004 * np.square(t) + 0.04 * t
        w_del = weights[:, np.newaxis] * delta
        # arr = arr + w_del
        return w_del

    if way == "gui":
        arr = np.array(gui_get_points())
    else:
        arr = get_circle()

    return pd.DataFrame(arr)


def generate_tsp(data, filename="assets/arr.tsp", comment="hello world"):
    """
    trans panda.DataFrame to .tsp file
    """
    info = {
        "NAME": os.path.basename(filename),
        "TIME": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "COMMENT": comment,
        "DIMENSION": data.shape[0],  # 行数
    }

    with open(filename, "w") as f:
        for key in info:
            f.write(key + " : " + str(info[key]) + "\n")
        f.write("NODE_COORD_SECTION\n")
        string = data.to_csv(
            path_or_buf=None,  # 直接写入文件的话，不知为何"EOF\n"的位置会出错
            sep=" ",
            na_rep="0",
            float_format="%.4f",
            header=False,
            mode="a+",
            line_terminator="\n")  # python 中\n\r 换两行！
        f.write(string)
        f.write("EOF\n")


def generate_tour(data,
                  filename="assets/tour.tour",
                  length=-1,
                  comment="hello world"):
    """
    data: [pandas.core.indexes.range.RangeIndex] list type works too!!
    trans index object to .tour file
    这个和下载到的.TOUR文件不是完全一样，不知道他们是怎么使用的，最后以-1结尾
    """
    info = {
        "NAME": os.path.basename(filename),
        "TIME": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "LENGTH": length,
        "COMMENT": comment,
        "DIMENSION": len(data),  # 行数
    }

    with open(filename, "w") as f:
        for key in info:
            f.write(key + " : " + str(info[key]) + "\n")
        f.write("TOUR_SECTION\n")
        f.write("\n".join([str(i) for i in list(data)]) + "\n")
        f.write("EOF\n")


def generate_obs(data, filename="assets/obs.obs", comment="hello world"):
    """
    trans DataFrame object to .obs file
    """
    info = {
        "NAME": os.path.basename(filename),
        "TIME": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "COMMENT": comment,
        "DIMENSION": data.shape[0],  # 行数
    }

    with open(filename, "w") as f:
        for key in info:
            f.write(key + " : " + str(info[key]) + "\n")
        f.write("OBSTACLE_SECTION\n")
        string = data.to_csv(
            path_or_buf=None,  # 直接写入文件的话，不知为何"EOF\n"的位置会出错
            sep=" ",
            na_rep="0",
            float_format="%.4f",
            header=False,
            mode="a+",
            line_terminator="\n")  # python 中\n\r 换两行！
        f.write(string)
        f.write("EOF\n")


if __name__ == "__main__":
    arr = generate_target(way="gui")
    obs = generate_target(way="gui")
    arr_plot(arr, obs)
    generate_tsp(arr)
    generate_obs(obs)
