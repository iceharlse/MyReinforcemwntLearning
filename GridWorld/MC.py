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

# 初始化参数
x = []  # 记录迭代次数
y = []  # 记录每次迭代的误差总和，达到一定数量级就停止训练
count = dict()  # 每个状态_动作对的次数，用来计算平均q值
qfunc = dict()  # 每个状态_动作对的q值
qfunc_last = dict()  # 用来记录上一次迭代的q值，用来运算误差

for s in states:
    for a in actions:
        qfunc["%d_%s" % (s, a)] = 0.0
        qfunc_last["%d_%s" % (s, a)] = 0.0
        count["%d_%s" % (s, a)] = 0.0001


# 计算误差
def computer_error(q_last, q):
    error = 0.0
    for key in q_last:
        error += q_last[key] - q[key]
    return error

#随机选择动作
def random_action():
    ran = int(random.random()*len(actions))
    return actions[ran]


#epsilon-greedy算法
def epsilon_greedy(s,epsilon):
    ran = random.random()
    if ran <=epsilon:
        return random_action()
    else:
        if pi[s] == '': return random_action()
        else:
            return pi[s]



# 开始迭代
#这里就能看到缺点了，他妈的要循环10000次才能找到合适的决策
for iter1 in range(10000):
    x.append(iter1)
    y.append(computer_error(qfunc_last, qfunc))
    qfunc_last = qfunc.copy()
    s_sample = []
    a_sample = []
    r_sample = []
    s = states[int(random.random() * len(states))]  # 随机初始状态
    terminate = False  # 终止标记
    i = 0

    # 得到一次episode的数据
    while terminate == False and i < 100:
        a = epsilon_greedy(s, 0.1)
        s1, r, terminate, _ = env.train_step(s, a)
        s_sample.append(s)
        r_sample.append(r)
        a_sample.append(a)
        s = s1
        i +=1

    #计算初始状态的报酬
    g = 0.0
    for i in range(len(s_sample)-1,-1,-1):
        g *= gamma
        g += r_sample[i]

    #得到初始状态的报酬后就可以求出后面所有的报酬了
    #   虽然我觉得这个可以直接用上一步一起算了，但是btw
    for i in range(len(s_sample)):
        key = "%d_%s"%(s_sample[i],a_sample[i])
        count[key] += 1.0
        qfunc[key] = (qfunc[key]*(count[key] - 1) + g)/ count[key]
        g -= r_sample[i]
        g /= gamma

        #策略改进
        if a_sample[i] == pi[s_sample[i]]:
            continue
        elif pi[s_sample[i]] == '':
            pi[s_sample[i]] = a_sample[i]
        else:
            key_pi = "%d_%s"%(s_sample[i],pi[s_sample[i]])
            if qfunc[key] > qfunc[key_pi]:
                pi[s_sample[i]] = a_sample[i]


#进行实际结果展示
for _ in range(100):
    state = env.reset()
    while state not in env.getTerminate_states():
        if state in env.getTerminate_states(): continue
        action = pi[state]
        state, r, is_terminal, info = env.step(action)
        env.render()
        time.sleep(0.1)