# 节点分配结果的可视化，显示各时刻下节点的位置、速度，并且会根据节点的发送个数进行不同速度的移动
# 输入：全局拓扑地图的节点列表数据、全局拓扑地图的边集数据、全局路径
# 关键输入：强化学习的结果，各机器人的及节点发送数量序列。

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import deque
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--RL_f', type=float, default=2)
parser.add_argument('--update_f', type=float, default=0.5)
parser.add_argument('--interval', type=int, default=500)

parser.add_argument('--node_info', default='config/sim_node.txt')
parser.add_argument('--line_info', default='config/match_node.txt')
parser.add_argument('--path_info', default='config/global_path.txt')
parser.add_argument('--send_result', default='config/send_result.txt')
args = parser.parse_args()

global mycolor
mycolor = ['orangered', 'dodgerblue', 'chocolate', 'darkturquoise', 'slategray', 'blueviolet',
           'royalblue', 'red', 'black', 'deepskyblue','darkgrey']
class node_date:
    def __init__(self):
        self.x = 0
        self.y = 0

class line_date:
    def __init__(self):
        self.p1 = 0
        self.p2 = 0

def LoadDate(path, type):
    f = open(path, "r")
    l_data = ''
    if type == 'node':
        node_list = {}
        while True:
            l_data = f.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                node = node_date()
                node.y = int(data[1])
                node.x = int(data[2])
                node_list[int(data[0])] = node
        return node_list
    else:
        line_list = []
        while True:
            l_data = f.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                line = line_date()
                line.p1 = int(data[0])
                line.p2 = int(data[3])
                line_list.append((line))
        return line_list

def Get_Globalroute(path):
    globa_path = []
    f = open(path, "r")
    l_data = ''
    while True:
        path_r = deque()
        l_data = f.readline()
        if not l_data:
            break
        else:
            data = l_data.split()
            for i in range(len(data)):
                path_r.append(int(data[i]))
            globa_path.append(path_r)
    return globa_path

def send_list_extension(path):
    scale = int(args.RL_f/args.update_f)
    send_list = []
    f = open(path, "r")
    l_data = ''
    while True:
        send_ = []
        l_data = f.readline()
        if not l_data:
            break
        else:
            data = l_data.split()
            for i in range(len(data)):
                temp_ = int(data[i])
                if scale==0:
                    send_.append(temp_)
                else:
                    for j in range(scale):
                        send_.append(temp_)
            send_list.append(send_)
    return send_list

class Route_Coordinates:
    def __init__(self, send_list, global_route, node_list, Robot_num):
        self.send_list=send_list
        self.global_route=global_route
        self.node_list=node_list
        self.Robot_num=Robot_num
        self.cor_x = []
        self.cor_y = []
        self.cor_fx = []
        self.cor_fy = []
        self.send_result = []
        self.frames=0

    def Calculate_main(self):
        # 各条路径的节点坐标序列
        pos = node_date()
        # 随机模式下，动作区间
        # send = [0, 1, 2]
        # 每个循环都是计算一个机器人的路径，几个机器人就循环几次
        for r in range(self.Robot_num):
            start = self.global_route[r][0]
            end = self.global_route[r][-1]
            x_r = []
            y_r = []
            n_fx = []
            n_fy = []
            send_r = []
            goal = self.node_list[end]
            pos= self.node_list[start]
            # pos.x = self.node_list[start].x
            # pos.y = self.node_list[start].y
            x_r.append(pos.x)
            y_r.append(pos.y)
            send_r.append(0)
            i = 0
            scale = 0
            a = len(self.send_list[r])
            # 计算r号机器人的路径点坐标
            while ((pos.x != goal.x or pos.y != goal.y) and i < a):

                # if self.send_list:
                #     # 训练模式（已经确定了发送个数）
                #     count = self.send_list[r][i]
                # else:
                #     # 随机生成发送序列
                #     count = random.choice(send)
                count = self.send_list[r][i]
                send_r.append(count)
                x_, y_, p_fx, p_fy, p, s = self.next_(self.global_route[r], count, pos, scale)
                pos = p
                scale = s
                x_r.append(x_)
                y_r.append(y_)
                n_fx.append(p_fx)
                n_fy.append(p_fy)
                i = i + 1
            self.send_result.append(send_r)
            self.cor_x.append(x_r)
            self.cor_y.append(y_r)
            self.cor_fx.append(n_fx)
            self.cor_fy.append(n_fy)

    def next_(self, global_path, send_nummber, p_c, scale):
        # 每次计算一个机器人下一时刻的节点坐标，输入节点列表、该机器人的全局路径，当前时刻的发送个数，当前时刻的节点位置，速度比例尺
        # 获取前向三节点信息
        n_1 = global_path.popleft()
        n_2 = global_path[0]
        n_3 = global_path[1]

        p_1 = self.node_list[n_1]
        p_2 = self.node_list[n_2]
        p_3 = self.node_list[n_3]

        f_x = []
        f_y = []
        if len(global_path) <= send_nummber or send_nummber==0:
            f_x.append(self.node_list[global_path[0]].x)
            f_y.append(self.node_list[global_path[0]].y)
        else:
            for i in range(send_nummber):
                f_x.append(self.node_list[global_path[i]].x)
                f_y.append(self.node_list[global_path[i]].y)

        delta = self.velocity_map(send_nummber)

        scale = scale + delta
        if scale >= 1:
            scale = scale - 1
            add_x = (p_3.x - p_2.x) * scale
            add_y = self.calculate_pose(p_2.x, p_2.y, p_3.x, p_3.y, add_x)
            point_x = (p_2.x + add_x)
            point_y = (p_2.y + add_y)
            p_c = p_2
        else:
            add_x = (p_2.x - p_1.x) * scale
            add_y = self.calculate_pose(p_1.x, p_1.y, p_2.x, p_2.y, add_x)
            point_x = (p_1.x + add_x)
            point_y = (p_1.y + add_y)
            p_c = p_1
            global_path.appendleft(n_1)

        return point_x, point_y, f_x, f_y, p_c, scale

    def velocity_map(self, num):
        v = num / 2
        detal = args.update_f * v
        return detal

    def calculate_pose(self, x1, y1, x2, y2, add_x):
        if add_x == 0:
            add_y = 0
        elif x1 - x2 == 0:
            add_y = (y1 - y2) * add_x
        else:
            add_y = ((y1 - y2) / (x1 - x2)) * add_x
        return add_y

    def Turn_Array(self):
        self.cor_x = self.align_length(self.cor_x)
        self.cor_y = self.align_length(self.cor_y)
        self.send_result = self.align_length(self.send_result)
        self.cor_x = np.array(self.cor_x)
        self.cor_y = np.array(self.cor_y)
        self.send_result = np.array(self.send_result)
        self.frames = len(self.send_result[0])
        return self.cor_x, self.cor_y, self.send_result,self.frames

    def align_length(self, list):
        max_l = 0
        for i in range(self.Robot_num):
            max_l = max(max_l, len(list[i]))
        for i in range(self.Robot_num):
            add = list[i][-1]
            l = max_l - len(list[i])
            while l > 0:
                list[i].append(add)
                l = l - 1
        return list

    def align_forward_route(self):
        for i in range(self.Robot_num):
            while len(self.cor_fx[i]) < self.frames:
                temp = self.cor_fx[i][-1]
                self.cor_fx[i].append(temp)
            while len(self.cor_fy[i]) < self.frames:
                temp = self.cor_fy[i][-1]
                self.cor_fy[i].append(temp)
        return self.cor_fx, self.cor_fy

class Robot_anim:
    def __init__(self, axis, r):
        (self.robot_,) = axis.plot([], [], "ro", markersize=20, color=mycolor[r], alpha=0.7)

    def set_graph_data(self, x, y):
        self.robot_.set_data(x, y)

        return (self.robot_,)

class Traj_anim:
    def __init__(self, axis, r):
        (self.traj_opt_,) = axis.plot([], [], color=mycolor[r], linestyle="dashed")

    def set_graph_data(self, x, y):
        self.traj_opt_.set_data(x, y)

        return (self.traj_opt_,)

class animation_show:
    def __init__(self,node_list, line_list, Robot_num):
        self.Robot_num=Robot_num
        self.node_list = node_list
        self.line_list = line_list

        self.fig = plt.figure(figsize = (15,10))
        self.ax = self.fig.add_subplot(111)

    def fig_set(self):
        point_x = []
        point_y = []

        for i in range(len(self.node_list)):
            point_x.append(self.node_list[i].x)
            point_y.append(self.node_list[i].y)

        ax_x = [point_x[0], point_x[-1]]
        ax_y = [point_y[0], point_y[-1]]
        ax_value = [ax_x[0] - 5, ax_x[1] + 20, ax_y[0] - 5, ax_y[1] + 5]

        self.ax.axis(ax_value)
        self.ax.invert_yaxis()
        self.ax.xaxis.set_ticks_position('top')

        plt.axis('off')

        for i in range(len(self.line_list)):
            a = [node_list[self.line_list[i].p1].x, node_list[self.line_list[i].p2].x]
            b = [node_list[self.line_list[i].p1].y, node_list[self.line_list[i].p2].y]
            plt.plot(a, b, '-', color=mycolor[10], alpha=0.2)

        for i in range(len(node_list)):
            plt.plot(point_x[i], point_y[i], "ro", markersize=20, alpha=0.2)
            plt.text(point_x[i] - 1, point_y[i] + 0.3, alpha=0.8, s='%d' % i)

    def func_anim_plot(self,ary_x,ary_y,ary_send,ary_fx,ary_fy,frame):
        self.ary_x = ary_x
        self.ary_y = ary_y
        self.ary_send = ary_send
        self.ary_fx = ary_fx
        self.ary_fy = ary_fy
        self.frame = frame

        self.robot_img = [Robot_anim(self.ax, _) for _ in range(self.Robot_num)]
        self.traj_img = [Traj_anim(self.ax, _) for _ in range(self.Robot_num)]

        ani = animation.FuncAnimation(self.fig, self._update_anim, interval=args.interval, frames=self.frame)

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.get_current_fig_manager().window.state('zoomed')
        save_to_video = False
        if save_to_video:
            # ani.save("222_g.mp4", writer="ffmpeg", dpi=300)
            ani.save("_g2.gif", writer="ffmpeg", dpi=300)
        else:
            plt.show()

    def update_forward_route(self, x, y, num, r):
        x_line = self.ary_fx[r][num]
        y_line = self.ary_fy[r][num]
        x_line = np.array(x_line)
        y_line = np.array(y_line)
        x_line = np.insert(x_line, 0, x)
        y_line = np.insert(y_line, 0, y)
        res = np.vstack((x_line, y_line))
        return res

    def _update_anim(self, num):

        if num >= len(self.ary_x[0]):
            x = self.ary_x[:, len(self.ary_x[0]) - 1]
            y = self.ary_y[:, len(self.ary_x[0]) - 1]
        else:
            x = self.ary_x[:, num]
            y = self.ary_y[:, num]

        for j in range(self.Robot_num):
            self.robot_img[j].set_graph_data(x[j], y[j])

        for j in range(self.Robot_num):
            path = self.update_forward_route(x[j], y[j], num, j)
            self.traj_img[j].set_graph_data(path[0][:], path[1][:])

if __name__ == '__main__':
    # 机器人个数
    Robot = 10
    # 加载节点信息
    node_list = LoadDate(args.node_info, 'node')
    line_list = LoadDate(args.line_info, 'line')
    global_route = Get_Globalroute(args.path_info)
    send_list = send_list_extension(args.send_result)

    animation_ = animation_show(node_list, line_list, Robot)
    animation_.fig_set()

    Traj_ = Route_Coordinates(send_list, global_route, node_list, Robot)
    Traj_.Calculate_main()
    ary_x, ary_y, ary_send, frame = Traj_.Turn_Array()
    ary_fx, ary_fy = Traj_.align_forward_route()

    animation_.func_anim_plot(ary_x, ary_y, ary_send, ary_fx, ary_fy, frame)
