import numpy as np
import tensorflow as tf
import time
from ARG_FILE import args
# from Network import Branch_Dqn, ReplayBuffer
from Network import Branch_Dqn, ReplayBuffer,ReplayBuffer_HER1,ReplayBuffer_HER2

class Agent:
    def __init__(self,env):
        self.env = env
        self.S_dim = self.env.observation_space.n
        self.A_dim = self.env.Robot
        self.A_pre_dim=self.env.action_space.n

        self.q_model = Branch_Dqn(self.S_dim, self.A_dim, self.A_pre_dim)
        self.t_model = Branch_Dqn(self.S_dim, self.A_dim, self.A_pre_dim)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.buffer_goal=ReplayBuffer_HER1()
        self.buffer_newgoal=ReplayBuffer_HER2()

        self.train_time=0

        current_time = time.strftime(('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.create_file_writer('log/' + current_time)

    def target_update(self):
        weights = self.q_model.model.get_weights()
        self.t_model.model.set_weights(weights)

    def replay(self):
        #第一种数据回放
        states, actions, rewards, next_states, done = self.buffer.sample()

        # #Q值计算(版本1)
        # #两种运算掩模
        # action_mask1 = tf.one_hot(actions, self.A_pre_dim)
        # action_mask2 = tf.one_hot(actions, self.A_pre_dim)
        #
        # target_q_values = self.t_model.predict(states)
        # target_q_values=tf.multiply(target_q_values, action_mask1)
        #
        # next_q_max_values = self.t_model.predict(next_states).max(axis=1)
        # #Q-learning更新公式
        # mean_max_value=rewards + tf.reduce_mean(next_q_max_values, axis=-1) * args.gamma * (1 - done)
        #
        # mean_max_value = tf.cast(mean_max_value, dtype=tf.float32)
        # mean_max_value = tf.reshape(tf.repeat(mean_max_value, self.A_dim * self.A_pre_dim), (-1, self.A_dim, self.A_pre_dim))
        # q_reality=tf.multiply(mean_max_value, action_mask2)
        #
        # target_q_values = tf.cast(target_q_values, dtype=tf.float32)
        # target=tf.add(target_q_values, q_reality)


        # Q值计算(版本2)
        action_mask1 = tf.cast(tf.one_hot(actions, self.A_pre_dim, 0, 1), dtype=tf.float32)
        action_mask2 = tf.cast(tf.one_hot(actions, self.A_pre_dim, 1, 0), dtype=tf.float32)

        #目标值(需要将对应action中的Q值更新)
        target=self.t_model.predict(states)
        target2 = self.t_model.predict(states)

        next_q_values = self.q_model.predict(next_states)
        next_q_max_action_index = tf.math.argmax(next_q_values, axis=2)

        next_q_max_action = tf.one_hot(next_q_max_action_index, self.A_pre_dim, axis=-1)
        target_q_values = self.t_model.predict(next_states)

        target_q_values=tf.multiply(target_q_values, next_q_max_action)
        mean_action = tf.reduce_mean(target_q_values, 2, keepdims=False)


        #Q-learning更新公式
        rewards = tf.reshape(rewards, [64, 1])
        rewards_= tf.cast(tf.repeat(rewards, self.A_dim, axis=1),dtype=tf.float32)

        gamma_ = tf.repeat(tf.reshape(tf.repeat(args.gamma, 64), [64, 1]), self.A_dim, axis=1)

        q_value = tf.add(rewards_, tf.multiply(mean_action, gamma_))

        q_reality = tf.reshape(tf.repeat(q_value, self.A_pre_dim), (-1, self.A_dim, self.A_pre_dim))
        target_q_update=tf.multiply(q_reality, action_mask2)

        target = tf.cast(target, dtype=tf.float32)
        target = tf.multiply(target, action_mask1)

        target=tf.add(target, target_q_update)

        self.q_model.train(states, target)
        loss=self.q_model.write_loss()[0]
        self.train_time=self.train_time+1
        # loss, td_error=self.q_model.train(states, target, action_mask1)
        return loss



    # def train(self, max_episodes=1000):
    #     learn_step_counter = 0
    #     goal = []
    #     for r in range(self.env.Robot):
    #         goal.append(self.env.global_path[r][-1])
    #     goal = np.array(goal)
    #     T_r = 0
    #     min_step = 100
    #     max_value = -200
    #     loss=10
    #     for ep in range(max_episodes):
    #         step=0
    #         done, total_reward = False, 0
    #         state = self.env.reset()
    #         # if learn_step_counter > 600:
    #         #    self.q_model.epsilon = 0.4
    #         # else:
    #         # self.q_model.epsilon = 1
    #         if learn_step_counter % 100 == 0:
    #             self.q_model.epsilon *= args.eps_decay
    #         while not done:
    #             step=step+1
    #             action = self.q_model.next_action(state, goal)
    #             next_state, reward, done, _ = self.env.step(action, state)
    #             self.buffer.put(state, action, reward*0.01, next_state, done)
    #             total_reward += reward
    #             state = next_state
    #             if step > 12:
    #                 break
    #         T_r += total_reward
    #         if self.buffer.size() >= args.batch_size:
    #             loss = self.replay()
    #             with self.writer.as_default():
    #                 tf.summary.scalar("loss", loss, step=self.train_time)
    #                 self.writer.flush()
    #         if learn_step_counter % args.replace_target_iter == 0:
    #             self.target_update()
    #         if ep != 0:
    #             with self.writer.as_default():
    #                 tf.summary.scalar("average_reword", T_r / ep, step=ep)
    #                 tf.summary.scalar("sum_reword", T_r, step=ep)
    #                 self.writer.flush()
    #         learn_step_counter += 1
    #         min_step=min(min_step,step)
    #         max_value=max(max_value,total_reward)
    #         print('EP{} time{} EpisodeReward={} min_step={} max_reword={} loss={}'.format(ep, step, total_reward, min_step, max_value,loss))
    #         with self.writer.as_default():
    #             tf.summary.scalar("reword", total_reward, step=ep)
    #             self.writer.flush()
    #     self.q_model.save_model()

    def train(self, max_episodes=1000):
        max_step=15
        goal=[]
        T_r=0
        min_step=15
        max_value=-100
        loss=10
        for r in range(self.env.Robot):
            goal.append(self.env.global_path[r][-1])
        goal=np.array(goal)
        for ep in range(max_episodes):
            done, total_reward, step = False, 0, 0
            state = self.env.reset(0)
            if ep % 200 == 0:
                self.q_model.epsilon *= args.eps_decay
            for step in range(max_step):
                action = self.q_model.next_action(state, goal)
                next_state, reward, done, _ = self.env.step(action, state)
                #直接放入经验池
                # self.buffer.put(state, action, reward*0.01, next_state, done)
                #利用HER经验池
                self.buffer_goal.add(state, action, reward*0.01, next_state, done, goal)
                total_reward += reward
                state = next_state
                if done:#如果已经到达目标，就跳出循环
                    break
            T_r += total_reward
            if step < 10:
                #对这个回合中的经验进行目标变换
                self.buffer_newgoal.add(self.buffer_goal , 4)
                #将两个经验集合放入最终的经验池中
                self.buffer.put(self.buffer_goal,self.buffer_newgoal)
                self.buffer_goal.clear()
                self.buffer_newgoal.clear()

            if self.buffer.size() >= args.batch_size:
                loss = self.replay()
                with self.writer.as_default():
                     tf.summary.scalar("loss", loss, step=self.train_time)
                     self.writer.flush()
            if ep % args.replace_target_iter == 0:
                self.target_update()
            if ep!=0 and ep % 10 == 0:
                with self.writer.as_default():
                    tf.summary.scalar("average_reword", T_r / ep, step=ep)
                    tf.summary.scalar("sum_reword", T_r, step=ep)
                    self.writer.flush()
            min_step=min(min_step, step)
            max_value=max(max_value, total_reward)

            print('EP{} time{} EpisodeReward={} min_step={} max_reword={} loss={}'.format(ep, step, total_reward*0.1, min_step, total_reward*0.1,loss))
            with self.writer.as_default():
                tf.summary.scalar("episode_reword", total_reward*0.1, step=ep)
                self.writer.flush()
        self.q_model.save_model()
