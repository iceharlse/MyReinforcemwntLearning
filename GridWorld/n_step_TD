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


def n_step_TD_prediction(env, n=4, alpha=0.4, num_of_episode=10, gamma=1):
    # n-step TD预测,返回state value表

    # 初始化V(S)
    value = np.zeros(env.nS)

    for episode in range(num_of_episode):
        env.reset()

        # 初始化状态集合S和收益集合R
        state_lst = [env.START_STATE]
        reward_lst = [0]

        # 总时间步
        t = 0
        T = float('inf')  # 结束时间步
        while True:
            t += 1

            # 采样并存储所有时间步下的reward和state
            if t < T:
                # 等概率采样并执行动作,得到下一步信息
                action = env.sample_action()
                next_state, reward, done = env.step(action)
                # 存储next_state和reward
                reward_lst.append(reward)
                state_lst.append(next_state)
                # 探索到terminal状态后就不再采样动作,第2部分代码块不会再执行
                if done:
                    T = t
            # 更新时间步，判断前n步有没有超过上下限
            update_t = t - n

            # 计算n-step内的累计reward和V(St+n)得到的回报returns，然后更新V(St)
            if update_t >= 0:
                returns = 0
                # 计算n-step之间的累计reward
                for time in range(update_t + 1, min(update_t + n, T) + 1):
                    returns += gamma ** (time - update_t - 1) * reward_lst[time]
                # 与V(St+n)累加得到完整returns
                if update_t + n < T:
                    returns += gamma ** n * value[state_lst[update_t + n]]
                # 3.3 更新V(St)
                value[state_lst[update_t]] += alpha * (returns - value[state_lst[update_t]])
            # 更新到terminal前一步则退出循环
            if update_t == T - 1:
                break
    return value


def get_average_error(env):
    TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
    TRUE_VALUE[0] = TRUE_VALUE[-1] = 0

    steps = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    num_of_episode = 10
    runs = 100
    gamma = 1

    errors = np.zeros((len(steps), len(alphas)))
    for run in range(runs):
        for step_i, n in enumerate(steps):
            for alpha_i, alpha in enumerate(alphas):
                value = n_step_TD_prediction(env, n, alpha, num_of_episode, gamma)
                errors[step_i, alpha_i] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / 19)
        print(
            "\r" + "#" * round((int(run) + 1) / runs * 60) + " " * (60 - round((int(run) + 1) / runs - 1)) + '|' + str(
                int(run) + 1) + '/' + str(runs), sep="", end="")
    errors /= runs
    return errors


def show(errors):
    steps = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('Average RMS error over 19 states and 10 episodes')
    plt.ylim([0.15, 0.65])
    plt.title('Performance of n-step TD methods as a fun of alpha')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()



env = Env(19)

errors = get_average_error(env)
show(errors)
