import random
import numpy as np
import matplotlib.pyplot as plt


class Env():
    # 构造一个环境类
    # 具体环境描述见 https://zhuanlan.zhihu.com/p/295098730

    def __init__(self, num_states=19):
        self.STATES = np.arange(1, num_states + 1)
        self.Terminal = [0, num_states + 1]
        self.START_STATE = (1 + num_states) // 2  # 19 to 10
        self.left, self.right = 0, 1
        self.action_space = [self.left, self.right]
        self.nS = num_states + 2  # 加上Terminal的两个状态即21个状态
        self.nA = 2
        self.state = self.START_STATE

    def reset(self):
        self.state = self.START_STATE
        return self.state

    def sample_action(self):
        # 等概率选择动作
        if random.random() > 0.5:
            return self.left
        else:
            return self.right

    def step(self, action):
        # 如果上一步已经到达终点，重新来过
        if self.state in self.Terminal:
            self.state = self.START_STATE

        done = False
        reward = 0
        # left-action
        if action == self.left:
            self.state -= 1
            if self.state in self.Terminal:
                done = True
                reward = -1
        # right-action
        else:
            self.state += 1
            if self.state in self.Terminal:
                done = True
                reward = 1
        return self.state, reward, done


def select_action_behavior_policy(action_value_set, epsilon):
    # 使用epsilon-greedy采样action
    prob = random.random()
    if prob > epsilon:
        action = np.argmax(action_value_set)
    else:
        action = random.randint(0, 1)
    return action


def n_step_Sarsa(env, n=4, alpha=0.4, epsilon_scope=[0.2, 0.05, 0.99], num_of_episode=100, gamma=1):
    # n-step Sarsa control,返回Q表
    epsilon = epsilon_scope[0]
    # 初始化Q表
    Q = np.zeros((env.nS, env.nA))
    for _ in range(num_of_episode):
        env.reset()
        # Init S_set, A_set and R_set and store S0, A0 and R0
        state_lst = [env.START_STATE]
        reward_lst = [0]
        action = select_action_behavior_policy(Q[env.START_STATE], epsilon)
        action_lst = [action]
        # 总时间步
        t = 0
        T = float('inf')
        while True:
            t += 1
            #采样并存储所有时间步下的reward和state
            if t < T:
                # 执行动作,得到下一步信息
                next_state, reward, done = env.step(action)
                # 存储next_state和reward
                reward_lst.append(reward)
                state_lst.append(next_state)
                # 2采样下一步动作,若探索到terminal状态后则不再采样动作,第2部分代码块不会再执行
                if done:
                    T = t
                else:
                    action = select_action_behavior_policy(Q[next_state], epsilon)
                    action_lst.append(action)
            # 更新时间步
            update_t = t - n
            # 计算n-step内的累计reward和Q(Sτ+n,Aτ+n)得到的回报returns，然后更新Q(Sτ,Aτ)
            if update_t >= 0:
                returns = 0
                # 计算n-step之间的累计reward
                for time in range(update_t + 1, min(update_t + n, T) + 1):
                    returns += gamma ** (time - update_t - 1) * reward_lst[time]
                # 3.2 与Q(Sτ+n,Aτ+n)累加得到完整returns
                if update_t + n < T:
                    returns += gamma ** n * Q[state_lst[update_t + n]][action_lst[update_t + n]]
                # 3.3 更新Q(Sτ,Aτ)
                Q[state_lst[update_t]][action_lst[update_t]] += alpha * (
                            returns - Q[state_lst[update_t]][action_lst[update_t]])
            # 更新到terminal前一步则退出循环
            if update_t == T - 1:
                break
        # 对epsilon进行衰减
        if epsilon >= epsilon_scope[1]: epsilon *= epsilon_scope[2]
    return Q


env = Env(19)
Q_table = n_step_Sarsa(env, n=4, alpha=0.4, epsilon_scope=[0.2, 0.05, 0.99], num_of_episode=100, gamma=1)
print(Q_table)
