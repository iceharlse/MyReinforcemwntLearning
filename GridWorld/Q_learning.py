import random
import gym
import time

env = gym.make('GridWorld-v0')
env.reset()
states = env.getStates()
actions = env.getAction()
list_null = ['' for _ in range(len(states))]
v = [0 for _ in range(len(states))]
pi = dict(zip(states, list_null))
gamma = env.getGamma()
alpha = 0.1

# 初始化参数
x = []  # 记录迭代次数
y = []  # 记录每次迭代的误差总和，达到一定数量级就停止训练
count = dict()  # 每个状态_动作对的次数，用来计算平均q值
qfunc = dict()  # 每个状态_动作对的q值

for s in states:
    for a in actions:
        qfunc["%d_%s" % (s, a)] = 0.0


# 随机选择动作
def random_action():
    ran = int(random.random() * len(actions))
    return actions[ran]


# epsilon-greedy算法
def epsilon_greedy(s, epsilon):
    ran = random.random()
    if ran <= epsilon:
        return random_action()
    else:
        if pi[s] == '':
            return random_action()
        else:
            return pi[s]


# 开始迭代
# 这里就能看到缺点了，他妈的要循环10000次才能找到合适的决策
for iter1 in range(10000):

    s = states[int(random.random() * len(states))]  # 随机初始状态
    a = epsilon_greedy(s, 0.1)  # 初始动作
    terminate = False  # 终止标记
    count = 0

    # 开始迭代
    while terminate == False and count < 100:
        s_a_now = "%d_%s" % (s, a)
        s_next, r, terminate, _ = env.train_step(s, a)
        s_a_next = ""
        q_max = -10

        for a_next in actions:
            if q_max < qfunc["%d_%s" % (s_next, a_next)]:
                q_max = qfunc["%d_%s" % (s_next, a_next)]
                s_a_next = "%d_%s" % (s_next, a_next)

        qfunc[s_a_now] = qfunc[s_a_now] + alpha * (r + gamma * qfunc[s_a_next] - qfunc[s_a_now])

        s = s_next
        a = epsilon_greedy(s , 0.1)
        count += 1


        # 策略改进
        for i in range(len(states)):
            q_now = qfunc["%d_%s" % (states[i], actions[0])]
            a_now = actions[0]
            for j in range(1,len(actions)):
                if q_now < qfunc["%d_%s" % (states[i], actions[j])]:
                    a_now = actions[j]
                    q_now = qfunc["%d_%s" % (states[i], actions[j])]

            if (pi[i+1] == '') or (qfunc["%d_%s" % (states[i], pi[i+1])] < q_now): pi[i+1] = a_now


# 进行实际结果展示
for _ in range(100):
    state = env.reset()
    while state not in env.getTerminate_states():
        if state in env.getTerminate_states(): continue
        action = pi[state]
        state, r, is_terminal, info = env.step(action)
        env.render()
        time.sleep(0.1)
