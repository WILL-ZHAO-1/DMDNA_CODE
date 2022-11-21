import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from collections import deque
import argparse
import math
import matplotlib.image as mpimg
from tensorflow.keras import models
from Myworld import Environment


parser = argparse.ArgumentParser()
parser.add_argument('--node_info', default='config/sim_node.txt')
parser.add_argument('--path_info', default='config/global_path.txt')
parser.add_argument('--line_info', default='config/match_node.txt')
parser.add_argument('--send_result', default='config/send_result.txt')
parser.add_argument('--send_result1', default='config/send_result1.txt')
args = parser.parse_args()

global mycolor
mycolor = ['orangered', 'darkturquoise', 'deepskyblue', 'chocolate', 'dodgerblue', 'blueviolet',
           'royalblue', 'red', 'slategray', 'black']

class node_date:
    def __init__(self):
        self.x = 0
        self.y = 0

class Obstacle:
    def __init__(self, x, y, size) -> None:
        self.x = x
        self.y = y
        self.size = size

class Path:
    def __init__(self, x, y, th, u_th, u_v) -> None:
        self.xs = x
        self.ys = y
        self.ths = th
        self.u_v = u_v
        self.u_th = u_th

def LoadDate(path):
    f = open(path, "r")
    l_data = ''
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
                send_.append(int(data[i]))
            send_list.append(send_)
    return send_list

def angle_range_corrector(angle):
    if angle > np.pi:
        while angle > np.pi:
            angle -= 2 * np.pi
    elif angle < -np.pi:
        while angle < -np.pi:
            angle += 2 * np.pi

    return angle

class Path_anim:
    def __init__(self, axis):
        (self.path_img,) = axis.plot(
            [], [], color="b", linestyle="dashed", linewidth=0.15
        )

    def set_graph_data(self, x, y):
        self.path_img.set_data(x, y)

        return (self.path_img,)

class Robot_anim:
    def __init__(self, axis, r):
        (self.robot_,) = axis.plot([], [], "ro", markersize=20, color=mycolor[r], alpha=0.7)

    def set_graph_data(self, x, y):
        self.robot_.set_data(x, y)

        return (self.robot_,)

class obstacles_anim:
    def __init__(self, axis,r):
        (self.pose_,) = axis.plot([], [], "ro", markersize=10, color=mycolor[9], alpha=0.7)

    def set_graph_data(self, x, y):
        self.pose_.set_data(x, y)

        return (self.pose_,)

class Traj_anim:
    def __init__(self, axis, r):
        (self.traj_,) = axis.plot([], [], "k", color=mycolor[r])

    def set_graph_data(self, x, y):
        self.traj_.set_data(x, y)

        return (self.traj_,)

class Traj_opt_anim:
    def __init__(self, axis, r):
        (self.traj_opt_,) = axis.plot([], [], color=mycolor[7], linestyle="dashed")

    def set_graph_data(self, x, y):
        self.traj_opt_.set_data(x, y)

        return (self.traj_opt_,)

class Animation_robot:
    def __init__(self, node_list, global_route, send_list, Robot_num):
        self.Robot_num = Robot_num
        self.node_list = node_list
        self.num = len(node_list)
        self.global_route = global_route
        self.send_list = send_list

        self.fig = plt.figure(figsize = (15,10))
        self.axis = self.fig.add_subplot(111)

    def fig_set(self):
        #坐标轴范围
        ax_x = [self.node_list[0].x, self.node_list[self.num - 1].x]
        ax_y = [self.node_list[0].y, self.node_list[self.num - 1].y]
        # ax_value = [ax_x[0] - 5, ax_x[1] + 20, ax_y[0] - 5, ax_y[1] + 5]
        ax_value = [0, ax_x[1] + 15, 0, ax_y[1] + 20]

        self.axis.axis(ax_value)
        self.axis.invert_yaxis()
        # self.axis.xaxis.set_ticks_position('top')
        self.axis.axes.get_yaxis().set_visible(False)
        self.axis.axes.get_xaxis().set_visible(False)

        # 显示全局路径
        # global_x = []
        # global_y = []
        # for i in range(self.Robot_num):
        #     point_x = []
        #     point_y = []
        #     for j in range(len(self.global_route[i])):
        #         point_x.append(self.node_list[self.global_route[i][j]].x)
        #         point_y.append(self.node_list[self.global_route[i][j]].y)
        #     global_x.append(point_x)
        #     global_y.append(point_y)

        global_x = []
        global_y = []
        for i in range(self.Robot_num):
            point_x = []
            point_y = []
            point_x.append(self.node_list[self.global_route[i][0]].x)
            point_y.append(self.node_list[self.global_route[i][0]].y)
            point_x.append(self.node_list[self.global_route[i][-1]].x)
            point_y.append(self.node_list[self.global_route[i][-1]].y)
            global_x.append(point_x)
            global_y.append(point_y)

        #画终点
        # for i in range(self.Robot_num):
        #     self.axis.plot(global_x[i][0], global_y[i][0], 'ro', markersize=15, color=mycolor[i])
        #     plt.text(global_x[i][0]+5, global_y[i][0] + 5, fontsize=25,alpha=0.8, s='robot-%d' % (i+1))
            # self.axis.plot(global_x[i][1], global_y[i][1], 'ro', markersize=10, color=mycolor[i])
            # self.axis.plot(global_x[i][-1], global_y[i][-1], "ro", markersize=15, color=mycolor[i], alpha=0.7)
        #
        # for i in range(len(global_x[0])):
        #     self.axis.plot(global_x[0][i], global_y[0][i], "ro", markersize=10, color=mycolor[i], alpha=0.7)

        # obstacles = [
        #     Obstacle(35, 25, 20),
        #     Obstacle(80, 45, 20),
        #     Obstacle(125, 85, 20),
        #     Obstacle(94, 100, 10),
        #     Obstacle(96, 120, 20),
        #     Obstacle(93, 169, 20),
        #     Obstacle(156, 150, 20),
        #     Obstacle(65, 180, 20),
        # ]
        # for obj in obstacles:
        #     self.axis.plot(obj.x, obj.y, "ro", markersize=10, color=mycolor[9], alpha=0.7)

        origin_map = mpimg.imread('map.png')  # 读取和代码处于同一目录下的 lena.png
        plt.imshow(origin_map,cmap='gray')  # 显示图片
        # plt.show()

    def show_result(self):
        for r in range(self.Robot_num):
            index_ = self.max_index[r]
            self.axis.plot(self.traj_x[r][index_], self.traj_y[r][index_],
            "ro", markersize=20, color=mycolor[r],alpha=0.7)
            self.axis.plot(self.traj_x[r][: index_ + 1], self.traj_y[r][: index_ + 1],
            "k", color=mycolor[r])

    def func_anim_plot(self, traj_x, traj_y, traj_th, traj_paths, traj_opt, obstacles, show_finall):
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.traj_th = traj_th
        self.traj_paths = traj_paths
        self.traj_opt = traj_opt
        self.max_path_num = 10
        self.obstacles = obstacles
        self.max_index = []
        for i in range(self.Robot_num):
            self.max_index.append(len(self.traj_opt[i]) - 1)

        if show_finall:
            self.show_result()
        else:
            self.robot_img = [Robot_anim(self.axis, _) for _ in range(self.Robot_num)]
            self.traj_img = [Traj_anim(self.axis, _) for _ in range(self.Robot_num)]
            self.traj_opt_img = [Traj_opt_anim(self.axis, _) for _ in range(self.Robot_num)]

            self.obstacles_img = [obstacles_anim(self.axis, _) for _ in range(1)]

            self.dwa_path = []
            for i in range(self.Robot_num):
                dwa_ = [Path_anim(self.axis) for _ in range(self.max_path_num)]
                self.dwa_path.append(dwa_)
            self.step_text = self.axis.set_title("")

            anim = ani.FuncAnimation(
                self.fig, self._update_anim, interval=10, frames=max(self.max_index)
            )

            save_to_video = False
            if save_to_video:
                # anim.save("Local_trajectory_generation.mp4", writer="ffmpeg", dpi=600)
                anim.save("_l_3.mp4", writer="ffmpeg", dpi=100)
                # anim.save("_l_obs.gif", writer="ffmpeg", dpi=300)
            else:
                plt.show()

    def _update_anim(self, num):

        x_ = []
        y_ = []
        for j in range(self.Robot_num):
            if num >= self.max_index[j]:
                index_ = self.max_index[j]
            else:
                index_ = num
            self.robot_img[j].set_graph_data(
                self.traj_x[j][index_], self.traj_y[j][index_]
            )
            for ob in self.obstacles:
                diff=pow(self.traj_x[j][index_]-ob.x,2)+pow(self.traj_y[j][index_]-ob.y,2)
                if (diff<=pow(20,2)):
                    x_.append(ob.x)
                    y_.append(ob.y)
                    # self.obstacles_img[0].set_graph_data(ob.x, ob.y)
        self.obstacles_img[0].set_graph_data(x_, y_)

        for j in range(self.Robot_num):
            if num >= self.max_index[j]:
                index_ = self.max_index[j]
            else:
                index_ = num
            self.traj_img[j].set_graph_data(
                self.traj_x[j][: index_ + 1], self.traj_y[j][: index_ + 1]
            )

        for j in range(self.Robot_num):
            if num >= self.max_index[j]:
                index_ = self.max_index[j]
            else:
                index_ = num
            self.traj_opt_img[j].set_graph_data(
                self.traj_opt[j][index_].xs, self.traj_opt[j][index_].ys
            )

        for k in range(self.max_path_num):
            for r in range(self.Robot_num):
                if num >= self.max_index[r]:
                    # index_ = self.max_index[index]
                    continue
                else:
                    index_ = num
                path_num = int(np.ceil(len(self.traj_paths[r][index_]) / (self.max_path_num)) * k)

                if path_num > len(self.traj_paths[r][index_]) - 1:
                    path_num = np.random.randint(0, len(self.traj_paths[r][index_]))

                self.dwa_path[r][k].set_graph_data(
                    self.traj_paths[r][index_][path_num].xs, self.traj_paths[r][index_][path_num].ys
                )

class TwoWheeledRobot:
    def __init__(self, init_x, init_y, init_th) -> None:
        self.x = init_x
        self.y = init_y
        self.th = init_th
        self.u_v = 0.0
        self.u_th = 0.0

        self.traj_x = [init_x]
        self.traj_y = [init_y]
        self.traj_th = [init_th]
        self.traj_u_v = [0.0]
        self.traj_u_th = [0.0]

    @staticmethod
    def state_equation(xi, u):
        dxi = np.empty(3)
        dxi[0] = u[1] * np.cos(xi[2])
        dxi[1] = u[1] * np.sin(xi[2])
        dxi[2] = u[0]
        return dxi

    def update_state(self, u_th, u_v, dt):
        self.u_th = u_th
        self.u_v = u_v

        xi_init = np.array([self.x, self.y, self.th])
        u = np.array([u_th, u_v])
        sol = solve_ivp(
            lambda t, xi: TwoWheeledRobot.state_equation(xi, u), [0, dt], xi_init
        )
        integrated = sol.y[:, -1]
        next_x = integrated[0]
        next_y = integrated[1]
        next_th = integrated[2]

        self.traj_x.append(next_x)
        self.traj_y.append(next_y)
        self.traj_th.append(next_th)

        self.x = next_x
        self.y = next_y
        self.th = next_th

        return self.x, self.y, self.th

class CoarseSimulator:
    def __init__(self) -> None:
        self.max_ang_acc = np.deg2rad(200)  # rad/s^2
        self.lim_max_ang_vel = np.pi  # deg/s
        self.lim_min_ang_vel = -self.lim_max_ang_vel

    def predict_state(self, ang_vel, vel, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []

        for _ in range(pre_step):
            x = vel * np.cos(th) * dt + x
            y = vel * np.sin(th) * dt + y
            th = ang_vel * dt + th

            next_xs.append(x)
            next_ys.append(y)
            next_ths.append(th)

        return next_xs, next_ys, next_ths

class ConstGoal:
    def __init__(self, node_list, global_route, velocity, R) -> None:
        self._x = []
        self._y = []
        self._vel = []
        for i in range(R):
            temp_x = []
            temp_y = []
            temp_vel = []
            for j, k in zip(global_route[i], velocity[i]):
                temp_x.append(node_list[j].x)
                temp_y.append(node_list[j].y)
                temp_vel.append(k)
            self._x.append(temp_x)
            self._y.append(temp_y)
            self._vel.append(temp_vel)
        self.iter = [1] * R

    def calc_goal(self, flag, x, y, v, r):
        if flag:
            t_v = self._vel[r][self.iter[r]]
            t_x = self._x[r][self.iter[r]]
            t_y = self._y[r][self.iter[r]]
            self.iter[r] = self.iter[r] + 1
            flag = False
        else:
            t_x = x
            t_y = y
            t_v = v

        return t_x, t_y, t_v, flag

class DWA:
    def __init__(self, samplingtime) -> None:
        self.simu_robot = CoarseSimulator()

        self.pre_step = 30

        self.delta_ang_vel = 0.1

        self.samplingtime = samplingtime

        self.weight_angle = 0.2
        self.weight_obs = 0.7

        area_dis_to_obs = 30
        self.area_dis_to_obs_sqrd = area_dis_to_obs ** 2

        score_obstacle = 10
        self.score_obstacle_sqrd = score_obstacle ** 2

        self.traj_paths = []
        self.traj_opt = []

    def calc_input(self, g_x, g_y, t_v, state, obstacles):
        paths = self._make_path(state, t_v)
        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles)
        self.traj_opt.append(opt_path)
        return paths, opt_path

    def _make_path(self, state, t_v):
        min_ang_vel, max_ang_vel = self._calc_range_vels(state)
        paths = []
        for ang_vel in np.arange(min_ang_vel, max_ang_vel, self.delta_ang_vel):
            vel = t_v
            next_x, next_y, next_th = self.simu_robot.predict_state(
                ang_vel,
                vel,
                state.x,
                state.y,
                state.th,
                self.samplingtime,
                self.pre_step,
            )
            paths.append(Path(next_x, next_y, next_th, ang_vel, vel))

        self.traj_paths.append(paths)

        return paths

    def _calc_range_vels(self, state):
        range_ang_vel = self.samplingtime * self.simu_robot.max_ang_acc

        min_ang_vel = max(state.u_th - range_ang_vel, self.simu_robot.lim_min_ang_vel)
        max_ang_vel = min(state.u_th + range_ang_vel, self.simu_robot.lim_max_ang_vel)

        return min_ang_vel, max_ang_vel

    def _eval_path(self, paths, g_x, g_y, state, obstacles):
        neighbor_obs = self._calc_neighbor_obs(state, obstacles)
        score_heading_angles = []
        score_obstacles = []

        for path in paths:
            score_obs = self._calc_obstacles_score(path, neighbor_obs)
            if score_obs == -float("inf"):
                continue
            score_obstacles.append(score_obs)
            score_heading_angles.append(self._calc_heading_angle_score(path, g_x, g_y))

        if len(score_heading_angles) == 0:
            raise RuntimeError("All paths cannot avoid obstacles")

        score_heading_angles_np = np.array(score_heading_angles)
        score_obstacles_np = np.array(score_obstacles)

        scores = (
                self.weight_angle * score_heading_angles_np
            + self.weight_obs * score_obstacles_np
        )
        return paths[scores.argmax()]

    def _calc_heading_angle_score(self, path, g_x, g_y):
        last_x = path.xs[-1]
        last_y = path.ys[-1]
        last_th = path.ths[-1]

        angle_to_goal = np.arctan2(g_y - last_y, g_x - last_x)
        score_angle = angle_to_goal - last_th

        score_angle = abs(angle_range_corrector(score_angle))

        score_angle = np.pi - score_angle

        return score_angle

    def _calc_heading_vel_score(self, path):
        return path.u_v

    def _calc_neighbor_obs(self, state, obstacles):
        neighbor_obs = []

        for obs in obstacles:
            temp_dis_to_obs = (state.x - obs.x) ** 2 + (state.y - obs.y) ** 2
            if temp_dis_to_obs < self.area_dis_to_obs_sqrd:
                neighbor_obs.append(obs)
        return neighbor_obs

    def _calc_obstacles_score(self, path, neighbor_obs):
        score_obstacle_sqrd = self.score_obstacle_sqrd
        for (path_x, path_y) in zip(path.xs, path.ys):
            for obs in neighbor_obs:
                temp_dis_to_obs = (path_x - obs.x) ** 2 + (path_y - obs.y) ** 2
                if temp_dis_to_obs < score_obstacle_sqrd:
                    score_obstacle_sqrd = temp_dis_to_obs
                if temp_dis_to_obs < obs.size + 0.75:  # マージン
                    return -float("inf")

        return np.sqrt(score_obstacle_sqrd)

class MainController:
    def __init__(self, node_list, global_route, velocity, Robot_num) -> None:
        self.samplingtime = 0.1
        self.Robot_num = Robot_num

        self.node_list = node_list
        self.global_route = global_route
        self.velocity = velocity
        self.robot = self.robot_init(self.Robot_num)
        self.goal_maker = ConstGoal(self.node_list, self.global_route, self.velocity, self.Robot_num)
        self.planner = self.dwa_init(self.samplingtime, self.Robot_num)

        self.obstacles = [
            Obstacle(35, 25, 20),
            Obstacle(80, 45, 20),
            Obstacle(125, 85, 20),
            Obstacle(94, 100, 10),
            Obstacle(96, 120, 20),
            Obstacle(93, 169, 20),
            Obstacle(166, 150, 20),
            Obstacle(65, 180, 20),
        ]


    def heading(self, x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1)

    def robot_init(self, R):
        # index = self.global_route[0][0]
        # index_ = self.global_route[0][1]
        # r = TwoWheeledRobot(self.node_list[index].x, self.node_list[index].y,
        #                     self.heading(self.node_list[index].x, self.node_list[index].y,
        #                                  self.node_list[index_].x, self.node_list[index_].y))
        robot = []
        for i in range(R):
            index = self.global_route[i][0]
            index_ = self.global_route[i][1]
            r = TwoWheeledRobot(self.node_list[index].x, self.node_list[index].y,
                                self.heading(self.node_list[index].x, self.node_list[index].y,
                                             self.node_list[index_].x, self.node_list[index_].y))
            robot.append(r)
        return robot

    def dwa_init(self, samplingtime, R):
        robot = []
        for i in range(R):
            planner = DWA(samplingtime)
            robot.append(planner)
        return robot

    def run(self):
        goal_th_sqrd = 5
        max_timestep = 1000
        for r in range(self.Robot_num):
            time_step = 0
            final_gx = self.node_list[self.global_route[r][-1]].x
            final_gy = self.node_list[self.global_route[r][-1]].y
            t_flag = True
            g_x = 0
            g_y = 0
            g_v = 0
            while True:
                g_x, g_y, g_v, t_flag = self.goal_maker.calc_goal(t_flag, g_x, g_y, g_v, r)

                _, opt_path = self.planner[r].calc_input(g_x, g_y, g_v, self.robot[r], self.obstacles)

                u_th = opt_path.u_th
                u_v = opt_path.u_v

                self.robot[r].update_state(u_th, u_v, self.samplingtime)

                dist_ = math.sqrt((g_x - self.robot[r].x) ** 2 + (g_y - self.robot[r].y) ** 2)

                dist_to_goal = math.sqrt((final_gx - self.robot[r].x) ** 2 + (final_gy - self.robot[r].y) ** 2)

                if dist_ < 5:
                    t_flag = True

                if dist_to_goal < goal_th_sqrd:
                    break
                time_step += 1
                if time_step >= max_timestep:
                    break

        Robot_traj_x = []
        Robot_traj_y = []
        Robot_traj_th = []
        traj_paths = []
        traj_opt = []
        for i in range(self.Robot_num):
            Robot_traj_x.append(self.robot[i].traj_x)
            Robot_traj_y.append(self.robot[i].traj_y)
            Robot_traj_th.append(self.robot[i].traj_th)
            traj_paths.append(self.planner[i].traj_paths)
            traj_opt.append(self.planner[i].traj_opt)

        return (
            Robot_traj_x,
            Robot_traj_y,
            Robot_traj_th,
            traj_paths,
            traj_opt,
            self.obstacles
        )

def main():

    Robot_num = 10
    env = Environment(args.node_info, args.line_info, args.path_info, args.send_result, Robot_num)
    node_list = LoadDate(args.node_info)
    global_route = Get_Globalroute(args.path_info)
    model = models.load_model('Modle/result.h5')
    env.reset(model)
    env.env.step()
    env.env.result()
    send_list = send_list_extension(args.send_result1)

    animation = Animation_robot(node_list, global_route, send_list, Robot_num)
    animation.fig_set()

    velocity = []
    for i in range(Robot_num):
        vel_ = [0] * len(global_route[i])
        index = 1
        for j in send_list[i]:
            for k in range(j):
                vel_[index] = j
                index += 1
        velocity.append(vel_)

    controller = MainController(node_list, global_route, velocity, Robot_num)
    (
        traj_x,
        traj_y,
        traj_th,
        traj_paths,
        traj_opt,
        obstacles
    ) = controller.run()

    animation.func_anim_plot(traj_x, traj_y, traj_th, traj_paths, traj_opt, obstacles, False)


if __name__ == "__main__":
    main()
