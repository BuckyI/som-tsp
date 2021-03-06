import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import tkinter as tk
import plot_data as p


def arr_plot(arr, obs, fbzs=None, savepath=None):
    "for testing useful for Dataframe and ndarray"

    def get_xy(arr):
        try:
            x = arr["x"].to_list()
            y = arr["y"].to_list()
        except AttributeError:
            x = arr[:, 0].tolist()
            y = arr[:, 1].tolist()
        return x, y

    data = get_xy(arr)
    plt.scatter(data[0], data[1], color="red", label="city")
    data = get_xy(obs)
    plt.scatter(data[0], data[1], color="black", label="obstacle")
    if fbzs is not None:
        for i in fbzs:
            fbz = np.row_stack((i, i[0]))  # 末尾添加开头
            plt.plot(fbz[:, 0], fbz[:, 1])
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath + "problem.png",
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=200)
    plt.show()


def generate(x_range=(0, 100), y_range=(0, 100)):
    """使用matplotlib一次性获取city 和obstacle，同步显示"""
    plt.ion()
    fig = plt.figure()
    axis = fig.subplots()
    axis.set_xlim(x_range[0], x_range[1])
    axis.set_ylim(y_range[0], y_range[1])

    axis.set_title("get city")
    city = []
    while True:
        data = axis.figure.ginput(1)
        if not data:
            break
        else:
            data = data[0]
        city.append(data)
        plt.scatter(data[0], data[1], marker="o", color="#6495ED", s=10)
    city = pd.DataFrame(city, columns=["x", "y"])

    axis.set_title("get obstacle")
    obs = []
    while True:
        data = axis.figure.ginput(1)
        if not data:
            break
        else:
            data = data[0]
        obs.append(data)
        plt.scatter(data[0], data[1], marker="x", color="#DC143C", s=10)
    obs = pd.DataFrame(obs, columns=["x", "y"])

    axis.set_title("get forbidden area")
    fbzs = []  # 含有ndarray的列表
    while True:
        fbz = []
        while True:
            data = axis.figure.ginput(1)
            if not data:
                break
            else:
                data = data[0]
            fbz.append(data)
            plt.scatter(data[0], data[1], marker="o", color="#FFB6C1", s=10)
        # 读取一个之后画出区域
        if len(fbz) == 0:
            break
        fbz = np.array(fbz)
        fbzs.append(fbz)
        cfbz = np.row_stack((fbz, fbz[0]))  # 末尾添加开头
        plt.plot(cfbz[:, 0], cfbz[:, 1])

    plt.close()
    plt.ioff()
    return city, obs, fbzs


def generate_target(x_range=(0, 100), y_range=(0, 100), way=None, info=""):
    """
    generate the travel target for the sales man
    return: [DataFrame] note y,x
    """
    def gui_get_points():
        """generate a gui window and click to get a list of points
        目前发现使用matplotlib更好，这个不需要了"""
        def callback(event):
            x, y = event.x, event.y
            var.set(str(x) + "," + str(y))
            print("\rclick position", x, y, end="")
            points.append((event.x, event.y))

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
        # arr = np.array(gui_get_points())
        arr = np.array(p.input_points(x_range, y_range, info))
    else:
        arr = get_circle()

    return pd.DataFrame(arr, columns=["x", "y"])


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

    data = data[["y", "x"]]  # 为了适应其他地方的tsp
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

    data = data[["y", "x"]]  # 为了适应其他地方的tsp

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


def generate_forbidden_zones(data, filename="assets/fbz.fbz"):
    with open(filename, "w") as f:
        for fbz in fbzs:
            string_ls = []
            for point in fbz:
                string_ls.append("{},{}".format(point[0], point[1]))
            f.write(";".join(string_ls) + "\n")


if __name__ == "__main__":
    time_id = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    path = "assets/" + "problem " + time_id + "/"
    os.mkdir(path)
    arr, obs, fbzs = generate()
    arr_plot(arr, obs, fbzs=fbzs, savepath=path)
    generate_tsp(arr, path + "arr.tsp")
    generate_obs(obs, path + "obs.obs")
    generate_forbidden_zones(fbzs, path + "fbz.fbz")
