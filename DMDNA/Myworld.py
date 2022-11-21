import random
import gym
import numpy
import numpy as np
from collections import deque
import math
from collections import deque

from gym import spaces

class node_date:
    def __init__(self):
        self.x = 0
        self.y = 0

class line_date:
    def __init__(self):
        self.p1 = 0
        self.p2 = 0

class Data_init:
    def __init__(self, node_info, line_info, path_info):
        self.node_list, self.line_list = self.LoadDate(node_info, line_info)
        self.globa_path = self.globapath(path_info)
        self.matrix = self.Creat_craph()
        self.Node_ = len(self.node_list)

    def LoadDate(self, path1, path2):
        node_list = {}
        line_list = []
        f1 = open(path1, "r")
        l_data = ''
        while True:
            l_data = f1.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                node = node_date()
                node.y = int(data[1])
                node.x = int(data[2])
                node_list[int(data[0])] = node
        f1.close()
        f1 = open(path2, "r")
        l_data = ''
        while True:
            l_data = f1.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                line = line_date()
                line.p1 = int(data[0])
                line.p2 = int(data[3])
                line_list.append((line))
        return node_list, line_list

    def globapath(self, g):
        globa_path = []
        f = open(g, "r")
        l_data = ''
        while True:
            path_r = []
            l_data = f.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                for i in range(len(data)):
                    path_r.append(int(data[i]))
                globa_path.append(path_r)
        return globa_path

    def Creat_craph(self):
        count = len(self.node_list)
        matrix = np.zeros((count, count), dtype=np.int)
        for line in self.line_list:
            i = line.p1
            j = line.p2
            matrix[i][j] = 1
            matrix[j][i] = 1
        return matrix

    def print_graph_info(self):
        node = len(self.node_list)
        line = len(self.line_list)
        print("...全局拓扑地图信息...")
        print("节点个数: %d" % node)
        print("边的个数：%d" % line)
        print("........")

class Env:
    def __init__(self, node_path, line_path, global_path, send_path, Robot):
        self.Robot = Robot
        self.node_list = {}
        self.line_list = []
        self.g_path = []
        self.globa_path = []
        self.LoadDate(node_path, line_path)
        self.get_globapath(global_path)
        self.send_list=[]
        self.send_path=send_path

    def LoadDate(self, path1, path2):
        f1 = open(path1, "r")
        l_data = ''
        while True:
            l_data = f1.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                node = node_date()
                node.y = int(data[1])
                node.x = int(data[2])
                self.node_list[int(data[0])] = node
        f1.close()
        f1 = open(path2, "r")
        l_data = ''
        while True:
            l_data = f1.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                line = line_date()
                line.p1 = int(data[0])
                line.p2 = int(data[3])
                self.line_list.append((line))

    def get_globapath(self, g):
        f = open(g, "r")
        l_data = ''
        while True:
            path_r = []
            l_data = f.readline()
            if not l_data:
                break
            else:
                data = l_data.split()
                for i in range(len(data)):
                    path_r.append(int(data[i]))
                self.g_path.append(path_r)

    def Creat_craph(self):
        count = len(self.node_list)
        matrix = np.zeros((count, count), dtype=np.int)
        for line in self.line_list:
            i = line.p1
            j = line.p2
            matrix[i][j] = 1
            matrix[j][i] = 1
        return matrix

    def print_graph_info(self):
        node = len(self.node_list)
        line = len(self.line_list)
        print("...全局拓扑地图信息...")
        print("节点个数: %d" % node)
        print("边的个数：%d" % line)
        print("........")

    def take_action(self, cur_state, next_state, action):
        crash = [1]*self.Robot
        crash = np.array(crash)
        while crash.all() == 1:
            # self.next_(action, next_state)
            for i in range(self.Robot):
                self.next_(action, next_state)
                for j in range(self.Robot):
                    if i==j:
                        continue
                    if next_state[i] == next_state[j]:
                        a=[i,j]
                        r = random.choice(a)
                        if action[r]>1:
                            action[r] = action[r] - 1
                    else:
                        flag, len_, r = self.forward_path_cut(i, j, action)
                        if flag:
                            action[r] = len_
                        else:
                            crash[i] = crash[j] = 0
        self.update_path(action, cur_state)
        return action, cur_state

    def next_(self, action, next_state):
        for i in range(self.Robot):
            path = self.globa_path[i]
            if action[i] >= len(path):
                next_state[i]=path[-1]
            else:
                next_state[i] = path[action[i]]

    def forward_path_cut(self, r1, r2, action):
        if action[r1]<=1 or action[r2]<=1:
            return False,-1,0
        path1 = []
        path2 = []
        g_path1 = self.globa_path[r1]
        g_path2 = self.globa_path[r2]
        for i in range(action[r1]+1):
            if i>=len(g_path1):
                path1.append(g_path1[-1])
            else:
                path1.append(g_path1[i])
        for i in range(action[r2]+1):
            if i>=len(g_path2):
                path2.append(g_path2[-1])
            else:
                path2.append(g_path2[i])

        flag = False
        for i in range(action[r1]+1):
            for j in range(action[r2]+1):
                if path1[i] == path2[j] and i==j:
                    flag = True
                    if i <= j:
                        len_ = j
                        return flag, len_, r2
                    else:
                        len_ = i - 1
                        return flag, len_, r1
        return flag, -1, 0

    def update_path(self, action, cur_state):
        for i in range(self.Robot):
            path = self.globa_path[i]
            if action[i] >= len(path):
                cur_state[i]=path[-1]
            else:
                for j in range(action[i]):
                    path.popleft()
                cur_state[i] = path[0]

    def mission_finish(self, cur_state, gola_):
        for i in range(self.Robot):
            if cur_state[i] != gola_[i]:
                return False
        return True

    def step(self):
        cur_state = [0] * self.Robot
        gola_ = [0] * self.Robot
        for i in range(self.Robot):
            cur_state[i] = self.g_path[i][0]
            gola_[i] = self.g_path[i][-1]

        self.globa_path.clear()
        for i in range(self.Robot):
            path_r = deque()
            for j in range(len(self.g_path[i])):
                path_r.append(self.g_path[i][j])
            self.globa_path.append(path_r)

        next_state = [0] * self.Robot
        while not self.mission_finish(cur_state, gola_):
            action = [0] * self.Robot
            for i in range(self.Robot):
                if cur_state[i] != gola_[i]:
                    action[i]=min(3,len(self.globa_path[i]))

            self.take_action(cur_state, next_state, action)
            self.send_list.append(action)


    def result(self):
        print(self.send_list)
        send=self.send_list
        w=[list() for i in range(self.Robot)]
        for i in range(len(send)):
            for j in range(self.Robot):
                w[j].append(send[i][j])

        t = ''
        with open(self.send_path, 'w') as q:
            # q.seek(0)
            # q.truncate()
            for i in w:
                for e in range(len(w[0])):
                    t = t + str(i[e]) + ' '
                q.write(t.strip(' '))
                q.write('\n')
                t = ''

class Environment():
    def __init__(self, node_info, line_info, path_info, send_path, num):
        self.data = Data_init(node_info, line_info, path_info)
        self.node_list = self.data.node_list
        self.global_path = self.data.globa_path
        self.state_num = len(self.node_list)
        self.matrix = self.data.matrix

        self.Robot = num
        self.update_path = []

        self.action_space = spaces.Discrete(3)  # [0,1,2]
        self.observation_space = spaces.Discrete(num)

        self.env = Env(node_info, line_info, path_info, send_path, num)

    def path_(self, action_r, r):
        path = self.update_path[r]
        if len(path) <= 2:
            return path[-1]
        for i in range(action_r):
            path.popleft()
        return path[0]

    def actionmap(self, num):  # action_space 是离散化的空间（0~n)
        return num
        # if num == 2:
        #     return 3
        # elif num == 1:
        #     return 2
        # else:
        #     return 1

    def step(self, action, pre_state):
        next_pos = []
        for i in range(self.Robot):
            action_ = self.actionmap(action[i])
            next_ = self.path_(action_, i)
            next_pos.append(next_)

        reword, done = self.Reward(action, next_pos, pre_state)
        self.state = np.array(next_pos)
        return self.state, reword, done, {}

    def Reward(self, action, next_pos, pre_state):
        # target_R_h = []
        target_R = 0
        target_r = 0
        done=[]

        for i in range(self.Robot):
            if (len(self.update_path[i])<=2):
                target_r=-10
                done.append(True)
            else:
                done.append(False)
                # target_r = (len(self.update_path[i]) - 2) * -1
                # target_r = math.pow(2, (len(self.update_path[i]) - 2))
                target_r = 5*(len(self.update_path[i]) - 2)
            target_R = target_R + target_r
            # target_R_h.append(target_r)
        # target_R = target_R / self.Robot

        done=np.array(done)
        if done.all():
            mission_R = 100
            task_done = True
        else:
            mission_R = -1
            task_done = False

        if task_done:
            Crash_P=0
        else:
            Crash_P = self.crash(action, next_pos, pre_state)

        total_reword = target_R + Crash_P + mission_R
        # total_reword = mission_R-target_R
        return total_reword, task_done

    def isconnect(self, i, j):
        if self.matrix[i][j] == 1:
            return True
        else:
            return False

    def crash(self, action, next_pos, pre_state):
        crash = numpy.zeros(self.Robot)
        E_dis = 0
        risk = 15
        save = 30
        # 判断节点间的碰撞可能(1.下一节点相同；2.速度相同下，交叉路径)
        for i in range(self.Robot):
            for j in range(i + 1, self.Robot):
                if next_pos[i] == next_pos[j]:
                    crash[i] = crash[j] = 1
                    break
                if action[i] == action[j] and self.isconnect(next_pos[i], next_pos[j]) \
                        and abs(next_pos[i] - pre_state[i]) != abs(next_pos[j] - pre_state[j]):
                        # and abs(next_pos[i] - pre_state[0][i]) != abs(next_pos[j] - pre_state[0][j]):
                    crash[i] = crash[j] = 1
                    break
                dis_ = self.caculate_dis(next_pos[i], next_pos[j])
                E_dis = E_dis + dis_
        E_dis = int(E_dis / (self.Robot * (self.Robot - 1) / 2))

        crash = numpy.array(crash)
        if crash.all() == 1:
            value = -10
        else:
            value = 0
            if E_dis > risk and E_dis < save:
                value = 10
            else:
                value = 20
            value = E_dis * 10

        return value

    def caculate_dis(self, point1, point2):
        node1 = self.node_list[point1]
        node2 = self.node_list[point2]
        dis = math.sqrt(pow((node1.x - node2.x), 2) + pow((node1.y - node2.y), 2))
        return int(dis)

    def reset(self, model):
        init_pos = []
        self.update_path.clear()
        for i in range(self.Robot):
            init_pos.append(self.global_path[i][0])
            path_r = deque()
            for j in range(len(self.global_path[i])):
                path_r.append(self.global_path[i][j])
            self.update_path.append(path_r)
        state = np.array(init_pos)
        return state

