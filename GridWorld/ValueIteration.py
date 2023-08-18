#值迭代算法
import gym
import time

env = gym.make('GridWorld-v0')
env.reset()
states = env.getStates()
actions = env.getAction()
list_null = ['' for _ in range(len(states))]
v = [0 for _ in range(len(states))]
pi = dict(zip(states,list_null))

#进行100次迭代保证收敛
for i in range(1000):
    env.reset()
    delta = 0.0

    for state in states:
        if state in env.getTerminate_states(): continue
        a1 = actions[0]
        next_state, r, is_terminal, info = env.train_step(state, a1)
        v1 = r + env.getGamma() * v[next_state-1]
        for action in actions:
            next_state, r, is_terminal, info = env.train_step(state, action)
            if v1 < r + env.getGamma() * v[next_state-1]:
                a1 = action
                v1 = r + env.getGamma() * v[next_state-1]
        delta += abs(v1 - v[state-1])
        pi[state] = a1
        v[state-1] = v1

    print('第%d次迭代' % i )

#进行实际结果展示
for _ in range(100):
    state = env.reset()
    while state not in env.getTerminate_states():
        if state in env.getTerminate_states(): continue
        action = pi[state]
        state, r, is_terminal, info = env.step(action)
        env.render()
        time.sleep(1)


