import numpy as np
import sys, random, time
from six import StringIO, b

from gym import utils, Env, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete

UP          = 0
LEFT        = 1
DOWN        = 2
RIGHT       = 3

MOVE        = 0
TURN_LEFT   = 1
TURN_RIGHT  = 2
PLACE_BLOCK = 3
MOVE_UP     = 0
MOVE_LEFT   = 1
MOVE_DOWN   = 2
MOVE_RIGHT  = 3
PLACE_UP    = 4
PLACE_LEFT  = 5
PLACE_DOWN  = 6
PLACE_RIGHT = 7
REWARD      = 1
TIMEOUT     = -.5
DIE         = -1

LAND        = 0
BRIDGE      = 1
GOAL        = 2
RIVER       = 3
DROWN       = 4
PLAYER      = 5
BLOCKED     = 9

RGB_LAND    = (122, 255, 122)
RGB_RIVER   = (  0, 204, 204)
RGB_BRIDGE  = (158,  79,   0)
RGB_GOAL    = (255, 226,  82)
RGB_PLAYER  = (255,  46,  46) # color used for indicating player's direction
PPB         = 9 # number of pixel along a axis per block


PLAYER_ARROW= np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
        ], dtype='uint8')

PLAYER_DROWN= np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
        ], dtype='uint')

# index for drawing player position
PLAYER_NI = np.nonzero(PLAYER_ARROW)
PLAYER_WI = np.nonzero(np.rot90(PLAYER_ARROW))
PLAYER_SI = np.nonzero(np.flipud(PLAYER_ARROW))
PLAYER_EI = np.nonzero(np.rot90(PLAYER_ARROW, 3))
PLAYER_D  = np.nonzero(PLAYER_DROWN)


MAPS = {
    '8x8_v' : [
        "SLLRLLLG",
        "LLLRLLLL",
        "LLLRLLLL",
        "LLLRLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
    ],
    '8x8_vcliff' : [
        "SLLRLLLG",
        "LLLRLLLL",
        "LLLRLLLL",
        "LLLRLLRR",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
    ],
    '8x8_ecliff' : [
        "SLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "RRRRLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "GLLLLLLL",
    ],
    '8x8_cliff' : [
        "SLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "RRRRLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLRLLLL",
        "GLLRLLLL",
    ],
    '8x8_blocked' : [
        "SLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "RRRRRRRR",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLRLLLL",
        "GLLRLLLL",
    ],
    '8x8' : [
        "SLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LRRRRRRR",
        "LRLLLLLL",
        "LRLLLLLL",
        "LRLLLLLL",
        "LRLLLLLG",
    ],
    '8x8_adv_1' : [
        "LLLLLLLS",
        "LLLLLLLL",
        "LLLLLLLL",
        "LLLLLLLL",
        "LRRRRRRR",
        "LRRRRRRR",
        "LRRLLLLL",
        "LRRLLLLG",
    ],
    '8x8_adv_2' : [
        "SLLLLLLL",
        "RRRRRRRL",
        "LLRLLLRL",
        "RLLLRLRL",
        "RRRRRLRL",
        "LLLLLLRL",
        "LRRRRRRR",
        "LLLLLLLG",
    ]
}

def rgb_to_gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

class BuildBridgeEnv(Env):
    metadata = {'render.modes' : ['human', 'ansi']}

    def __init__(self, desc=None, map_name=None, use_random_map=False,
                 use_time_limit=True, time_limit=400, step_penalty=-0.00125,
                 step_render=False, no_drown=False, gray_scale=False):

        if desc is None and map_name is None and use_random_map is False :
            return ValueError("Must provide either desc, map_name, or random flag")
        elif use_random_map is True :
            self.desc = self._random_map()
        elif desc is None :
            self.desc = np.asarray(MAPS[map_name], dtype='c')

        self.built_done = False
        self.use_random_map = use_random_map
        self.time_limit = time_limit
        self.use_time_limit = use_time_limit
        self.step_penalty = step_penalty
        self.nrow, self.ncol = self.desc.shape
        self.step_render = step_render
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(PPB*self.nrow*PPB*self.ncol)
        self.observation_space.shape = (PPB*self.nrow, PPB*self.ncol)
        self.global_step = 0
        self.step_cnt = 0
        self.spec = None
        self.no_drown = no_drown
        self.gray_scale = gray_scale

        self._reset()

    def _reset(self) :

        if self.use_random_map :
            self.desc = self._random_map()
            self.map, self.image, self.start_pos, self.goal_pos = self._build_map()
        elif self.built_done is False :
            self.map, self.image, self.start_pos, self.goal_pos = self._build_map()
            self._map = self.map.copy()
            self._image = self.image.copy()
            self.built_done = True
        else :
            self.map = self._map.copy()
            self.image = self._image.copy()

        self.step_cnt = 0
        self.player_dir = np.random.randint(4)
        self.player_pos = self.orig_block = self.start_pos
        self.cur_desc = self.desc.copy()

        self.draw_player(self.player_pos[0], self.player_pos[1])

        return self._get_observation()[0]

    def _step(self, a) :
        if   a == MOVE        : self._move_forward()
        elif a == TURN_LEFT   : self._turn_dir(1)
        elif a == TURN_RIGHT  : self._turn_dir(-1)
        elif a == PLACE_BLOCK : self._place_block()

        self.global_step += 1
        self.step_cnt += 1


        return self._get_observation()

    def _get_observation(self) :

        state = self.image if self.gray_scale is False else rgb_to_gray(self.image)
        r, c = self.player_pos

        if self.player_pos == self.goal_pos :
            done   = True
            reward = REWARD
        elif self.step_cnt > self.time_limit :
            done   = True
            reward = TIMEOUT
        elif self.map[r, c] == RIVER:
            done   = True
            reward = DIE
        else :
            done   = False
            reward = self.step_penalty

        if self.step_render :
            self._render(self.mode)

        return state, reward, done, {}

    def _render(self, mode='human', close=False):
        if close : return

        row, col = self.player_pos
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.cur_desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        outfile.write("\nDirection : {}, Action Count : {}({})\n"
                .format(["UP", "LEFT", "DOWN", "RIGHT"][self.player_dir], self.step_cnt, self.global_step))
        outfile.write("\n".join(''.join(line) for line in desc)+'\n')

        if mode != 'human' : return outfile

    def _turn_dir(self, direction):
        r, c = self.player_pos
        self.player_dir = (self.player_dir + direction) % 4
        self.draw_player(r, c, r, c)

    def _move_forward(self):
        r, c = self._get_front(self.player_pos, self.player_dir)
        if self.no_drown and (self.map[r, c] == RIVER):
            return self.player_pos

        old_r, old_c = self.player_pos
        self.player_pos = r, c
        self.draw_player(r, c, old_r, old_c)

    def _get_front(self, pos, direction):
        return self._get_dir_dist(pos, direction, 1)

    def _get_dir_dist(self, pos, direction, dist):
        r, c = pos

        if   direction == 0: r = np.maximum(r - dist, 0)
        elif direction == 1: c = np.maximum(c - dist, 0)
        elif direction == 2: r = np.minimum(r + dist, self.nrow - 1)
        elif direction == 3: c = np.minimum(c + dist, self.ncol - 1)

        return r, c

    def _place_block(self) :
        r, c = self._get_front(self.player_pos, self.player_dir)
        if self.map[r, c] == RIVER :
            self.map[r, c] = BRIDGE
            self.cur_desc[r, c] = 'B'
            self.draw_bridge(r, c)

    def _build_map(self) :
        m = np.zeros((self.nrow, self.ncol), dtype='int')
        i = np.zeros((self.nrow*PPB, self.ncol*PPB, 3), dtype='uint8')
        sp = (0, 0)
        gp = (0, 0)

        for r in range(self.nrow) :
            _r = r * PPB
            for c in range(self.ncol) :
                _c = c * PPB

                if   self.desc[r, c] == b'S' :
                    sp = r, c
                    i[_r:_r+PPB, _c:_c+PPB] = RGB_LAND
                elif self.desc[r, c] == b'B' :
                    m[r, c] = BRIDGE
                    i[_r:_r+PPB, _c:_c+PPB] = RGB_BRIDGE
                elif self.desc[r, c] == b'R' :
                    m[r, c] = RIVER
                    i[_r:_r+PPB, _c:_c+PPB] = RGB_RIVER
                elif self.desc[r, c] == b'G' :
                    m[r, c] = GOAL
                    gp = r, c
                    i[_r:_r+PPB, _c:_c+PPB] = RGB_GOAL
                else :
                    i[_r:_r+PPB, _c:_c+PPB] = RGB_LAND

        return m, i, sp, gp


    def _random_map(self):
        return np.asarray(random.choice(list(MAPS.values())), dtype='c')

    def draw_player(self, r, c, old_r=None, old_c=None):

        if old_r is not None and old_c is not None :
            orig_block = self.map[old_r, old_c]
            if orig_block == LAND :
                self.image[old_r*PPB:(old_r+1)*PPB, old_c*PPB:(old_c+1)*PPB] = RGB_LAND
            elif orig_block == BRIDGE :
                self.image[old_r*PPB:(old_r+1)*PPB, old_c*PPB:(old_c+1)*PPB] = RGB_BRIDGE

        block = self.image[r*PPB:(r+1)*PPB, c*PPB:(c+1)*PPB]

        if   self.map[r, c] == RIVER  : block[PLAYER_D] = RGB_PLAYER
        elif self.player_dir == UP    : block[PLAYER_NI] = RGB_PLAYER
        elif self.player_dir == LEFT  : block[PLAYER_WI] = RGB_PLAYER
        elif self.player_dir == DOWN  : block[PLAYER_SI] = RGB_PLAYER
        elif self.player_dir == RIGHT : block[PLAYER_EI] = RGB_PLAYER

    def draw_bridge(self, r, c) :
        self.image[r*PPB:(r+1)*PPB, c*PPB:(c+1)*PPB] = RGB_BRIDGE

if __name__ == '__main__':

    BBE = BuildBridgeEnv(use_random_map=True)

    import matplotlib.pyplot as plt
    import cv2


    plt.ion()

    s = BBE.reset()
    d = False
    while True :
        BBE.render()
        key = input()
        if d :
            print("done!")
            s = BBE.reset()

        if   key == 'w' :
            s, r, d, _ = BBE.step(MOVE)
            print("Action : Move")
        elif key == 'l' :
            s, r, d, _ = BBE.step(TURN_LEFT)
            print("Action : Turn Left")
        elif key == 'r' :
            s, r, d, _ = BBE.step(TURN_RIGHT)
            print("Action : Turn Right")
        elif key == 'b' :
            s, r, d, _ = BBE.step(PLACE_BLOCK)
            print("Action : Place bridge")
        elif key == 'e' : break
        else : continue
        x = rgb_to_gray(s)
        plt.imshow(x, cmap='gray')



