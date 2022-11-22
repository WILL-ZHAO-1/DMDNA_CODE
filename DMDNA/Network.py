import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random
from ARG_FILE import args

class MDN:
    def __init__(self, state_dim, action_dim, action_pre_dim):
        self.S_dim = state_dim
        self.A_dim = action_dim
        self.A_pre_dim = action_pre_dim
        self.epsilon = args.eps
        self.optimizer = Adam(learning_rate=args.lr, clipnorm=1)
        self.model = self.create_model()
        self.loss=0
        self.td_error=0

    def create_model(self):
        # input
        state_input = Input(self.S_dim, name='Input')
        out = state_input

        # The shared network
        for i in range(1):
            out = Dense(args.net_share, activation='relu', name='common{}'.format(i))(out)

        # The state_value
        state_value = out
        state_value = Dense(args.net_state, activation='relu', name='state_value{}'.format(1))(state_value)
        state_value = Dense(1, activation='relu', name='state_value')(state_value)

        # The subaction_network
        sub_action = []
        for i in range(self.A_dim):
            sub_act_out = out
            for j in range(1):
                sub_act_out = Dense(args.net_action, activation='relu', name='sub_action{}-{}'.format(i, j))(sub_act_out)
            sub_act_out = Dense(self.A_pre_dim, activation=None, name='sub_action_out{}'.format(i))(sub_act_out)
            sub_action.append(sub_act_out)


        total_action = tf.stack(sub_action, axis=1)
        mean_action = tf.reduce_mean(total_action, 1, keepdims=True)
        var_action = tf.subtract(total_action, mean_action)

        action_output = Add()([state_value, var_action])
        model = tf.keras.Model(state_input, action_output)

        model.compile(loss=args.loss, optimizer=Adam(args.lr))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def next_action(self, state, goal):
        state = np.reshape(state, [1, self.S_dim])
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        next_action = []
        if np.random.random() < self.epsilon:
            for i in range(self.A_dim):
                if (state[0][i]==goal[i]):
                    next_action.append(0)
                else:
                    next_action.append(random.randint(0, self.A_pre_dim - 1))
        else:
            for i in range(self.A_dim):
                if (state[0][i]==goal[i]):
                    next_action.append(0)
                else:
                    next_action.append(np.argmax(q_value[i]))
        next_action = np.array(next_action)
        return next_action

    def train(self, states, targets):
        # return self.my_train(states, targets, action_mask)
        self.history=self.model.fit(states, targets, epochs=1, verbose=0)


    def save_model(self, file_path=args.save_path):
        print('model saved')
        self.model.save(file_path)

    def write_loss(self,):
        return self.history.history['loss']

class ReplayBuffer:
    def __init__(self, capacity=args.capacity):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

# class ReplayBuffer:
#     def __init__(self, capacity=args.capacity):
#         self.buffer = deque(maxlen=capacity)
#
#     def put(self, buffer1, buffer2):
#         for index in range(buffer1.size()):
#             self.buffer.append(buffer1.buffer[index])
#         for index in range(buffer2.size()):
#             self.buffer.append(buffer2.buffer[index])
#
#     def sample(self):
#         sample = random.sample(self.buffer, args.batch_size)
#         states, actions, rewards, next_states, done, _ = map(np.asarray, zip(*sample))
#         states = np.array(states).reshape(args.batch_size, -1)
#         next_states = np.array(next_states).reshape(args.batch_size, -1)
#         return states, actions, rewards, next_states, done
#
#     def size(self):
#         return len(self.buffer)

class ReplayBuffer_HER1:
    def __init__(self):
        self.buffer = []

    def add(self, state, action, reward, next_state, done, goal):
        self.buffer.append((state, action, reward, next_state, done, goal))

    def clear(self):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)


class ReplayBuffer_HER2:
    def __init__(self):
        self.buffer = []

    def add(self, ep_buffer, sample,):
        if ep_buffer.size() > 10:
            return
        else:
            for t in range(ep_buffer.size()):
                for i in range(sample):
                    future = np.random.randint(t, ep_buffer.size())
                    goal = ep_buffer.buffer[future][3]  # next_state of future
                    state = ep_buffer.buffer[t][0]
                    action = ep_buffer.buffer[t][1]
                    next_state = ep_buffer.buffer[t][3]
                    done=(next_state == goal).all()
                    if done:
                        reward =10
                        self.buffer.append((state, action, reward*0.01, next_state, done, goal))
    def clear(self):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)