import logging
import numpy
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        self.states = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] #状态空间
        self.x=[275,325,375,425,475,275,325,375,425,475,275,325,375,425,475,275,325,375,425,475,275,325,375,425,475]
        self.y=[475,475,475,475,475,425,425,425,425,425,375,375,375,375,375,325,325,325,325,325,275,275,275,275,275]
        self.terminate_states = dict()  #终止状态为字典格式
        self.terminate_states[15] = 1

        self.actions = ['n','e','s','w']

        self.rewards = dict();        #回报的数据结构为字典
        self.rewards[4] = -1.0
        self.rewards[9] = -1.0
        self.rewards[11] = -1.0
        self.rewards[12] = -1.0
        self.rewards[15] = 1.0
        self.rewards[23] = -1.0
        self.rewards[24] = -1.0
        self.rewards[25] = -1.0

        self.t = dict();             #状态转移的数据格式为字典
        #第一行
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_s'] = 7
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 8
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_s'] = 9
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 10
        self.t['5_w'] = 4
        #第二行
        self.t['6_n'] = 1
        self.t['6_s'] = 11
        self.t['6_e'] = 7
        self.t['7_n'] = 2
        self.t['7_s'] = 12
        self.t['7_w'] = 6
        self.t['7_e'] = 8
        self.t['8_n'] = 3
        self.t['8_s'] = 13
        self.t['8_w'] = 7
        self.t['8_e'] = 9
        self.t['9_n'] = 4
        self.t['9_s'] = 14
        self.t['9_w'] = 8
        self.t['9_e'] = 10
        self.t['10_n'] = 5
        self.t['10_s'] = 15
        self.t['10_w'] = 9
        #第三行
        self.t['11_n'] = 6
        self.t['11_s'] = 16
        self.t['11_e'] = 12
        self.t['12_n'] = 7
        self.t['12_s'] = 17
        self.t['12_w'] = 11
        self.t['12_e'] = 13
        self.t['13_n'] = 8
        self.t['13_s'] = 18
        self.t['13_w'] = 12
        self.t['13_e'] = 14
        self.t['14_n'] = 9
        self.t['14_s'] = 19
        self.t['14_w'] = 13
        self.t['14_e'] = 15
        self.t['15_n'] = 10
        self.t['15_s'] = 20
        self.t['15_w'] = 14
        #第四行
        self.t['16_n'] = 11
        self.t['16_s'] = 21
        self.t['16_e'] = 17
        self.t['17_n'] = 12
        self.t['17_s'] = 22
        self.t['17_w'] = 16
        self.t['17_e'] = 18
        self.t['18_n'] = 13
        self.t['18_s'] = 23
        self.t['18_w'] = 17
        self.t['18_e'] = 19
        self.t['19_n'] = 14
        self.t['19_s'] = 24
        self.t['19_w'] = 18
        self.t['19_e'] = 20
        self.t['20_n'] = 15
        self.t['20_s'] = 25
        self.t['20_w'] = 19
        #第五行
        self.t['21_n'] = 16
        self.t['21_e'] = 22
        self.t['22_n'] = 17
        self.t['22_w'] = 21
        self.t['22_e'] = 23
        self.t['23_n'] = 18
        self.t['23_w'] = 22
        self.t['23_e'] = 24
        self.t['24_n'] = 19
        self.t['24_w'] = 23
        self.t['24_e'] = 25
        self.t['25_n'] = 20
        self.t['25_w'] = 24


        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions
    def getTerminate_states(self):
        return self.terminate_states
    def setAction(self,s):
        self.state=s

    def step(self, action):
        #系统当前状态
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值

        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if self.state not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[self.state]


        return next_state, r, is_terminal, {}

    def train_step(self, state, action):

        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s"%(state, action)   #将状态和动作组成字典的键值

        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if self.state not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[self.state]


        return next_state, r, is_terminal, {}


    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 800
        screen_height = 600

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建网格世界

            # 12根线rendering.Line按照右坐标到横坐标给值，坐标定值时按照（纵，横）给值
            self.line1 = rendering.Line((500, 500), (250, 500))
            self.line2 = rendering.Line((500, 450), (250, 450))
            self.line3 = rendering.Line((500, 400), (250, 400))
            self.line4 = rendering.Line((500, 350), (250, 350))
            self.line5 = rendering.Line((500, 300), (250, 300))
            self.line6 = rendering.Line((500, 250), (250, 250))
            self.line7 = rendering.Line((500, 500), (500, 250))
            self.line8 = rendering.Line((450, 500), (450, 250))
            self.line9 = rendering.Line((400, 500), (400, 250))
            self.line10 = rendering.Line((350, 500), (350, 250))
            self.line11 = rendering.Line((300, 500), (300, 250))
            self.line12 = rendering.Line((250, 500), (250, 250))

            # 给线条一个颜色
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)

            # 把线段添加在容器里面
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)

            # 创建墙体
            # 创建第一个黑色区域,这是按照右下，右上、左上、左下给的值
            self.heikuai_1 = rendering.FilledPolygon([(450, 400), (450, 500), (400, 500), (400, 400)])
            self.heikuai_1.set_color(0, 0, 0)
            self.viewer.add_geom(self.heikuai_1)

            # 创建第二个黑色区域,这是按照右下，右上、左上、左下给的值
            self.heikuai_2 = rendering.FilledPolygon([(350, 350), (350, 400), (250, 400), (250, 350)])
            self.heikuai_2.set_color(0, 0, 0)
            self.viewer.add_geom(self.heikuai_2)

            # 创建第三个黑色区域,这是按照右下，右上、左上、左下给的值
            self.heikuai_3 = rendering.FilledPolygon([(500, 250), (500, 300), (350, 300), (350, 250)])
            self.heikuai_3.set_color(0, 0, 0)
            self.viewer.add_geom(self.heikuai_3)

            # 创建出口--圆
            self.ChuKou = rendering.make_circle(10)
            self.transform = rendering.Transform(translation=(475, 375))
            self.ChuKou.add_attr(self.transform)
            self.ChuKou.set_color(0, 0, 0)
            self.viewer.add_geom(self.ChuKou)
            #创建机器人
            self.robot= rendering.make_circle(5)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(self.robot)


        if self.state is None: return None
        #self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])



        return self.viewer.render(return_rgb_array=mode == 'rgb_array')