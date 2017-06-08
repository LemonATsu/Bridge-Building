import numpy as np
import sys, random, time
from six import StringIO, b

from gym import utils, Env, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete

MOVE        = 0
TURN_LEFT   = 1
TURN_RIGHT  = 2
PLACE_BLOCK = 3
REWARD      = 1000
TIMEOUT     = -1000
DIE         = -1000

BRIDGE      = 1
GOAL        = 2
RIVER       = 3
DROWN       = 4
PLAYER      = 5

MAPS = {
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
        "RRRRRRRL",
        "RRRRRRRL",
        "LLLLLRRL",
        "LLLLLRRL",
        "GLLLLRRL",
        "RRRRRRRL",
    ],
    '8x8_adv_2' : [
        "SLLLLLLL",
        "RRRRRRRL",
        "GLRLLLRL",
        "RLLLRLRL",
        "RRRRRLRL",
        "LLLLLLRL",
        "LRRRRRRL",
        "LLLLLLLL",
    ]
}


class BuildBridgeEnv(Env):
    """
    S : starting point
    L : land
    B : bridge
    R : river, cannot pass it unless you have built bridge
    G : goal
    """

    metadata = {'render.modes' : ['human', 'ansi']}


    def __init__(self, desc=None, map_name=None, step_penalty=0, use_random_map=False, give_details=False, put_player=True,
                    use_time_limit=False, time_limit=40, use_flatten=False, step_render=False):
        """
        desc           : map description, as shown in MAPS var.
        map_name       : predefined map
        step_penalty   : penalty received after each step you have take
        use_random_map : randomly select map from MAPS
        give_details   : return the exact position/direction of player.
                         use_flatten must be set to True when using this.
        put_player     : put player on the returned state(map)
        use_time_limit : limit the number of step(time) the player can take
        time_limit     : the amount of step that a player can take when
                         use_time_limit is set to True
        step_render    : render after each step
        """

        if desc is None and map_name is None and use_random_map is False:
            raise ValueError('Must provide either desc or map_name')
        elif desc is not None :
            self.desc = desc = np.asarray(desc, dtype='c')
        elif use_random_map is True:
            self.desc = desc = self._random_map()
        elif desc is None:
            self.desc = desc = np.asarray(MAPS[map_name], dtype='c')

        if give_details and use_flatten is False :
            raise ValueError('Must set use_flatten to true when you want to use give_details')

        self.built_done        = False
        self.time_limit        = time_limit
        self.use_time_limit    = use_time_limit
        self.use_random_map    = use_random_map
        self.step_penalty      = step_penalty
        self.nrow, self.ncol   = nrow, ncol = desc.shape
        self.action_space      = spaces.Discrete(4)
        self.use_flatten       = use_flatten
        self.give_details      = give_details
        self.put_player        = put_player
        self.total_step        = 0
        self._best             = -100
        self.step_render       = step_render

        if give_details is False :
            self.observation_space = spaces.Discrete(nrow * ncol)
            self.observation_space.shape = (nrow, ncol) if use_flatten is False else (nrow * ncol, )
        else :
            self.observation_space = spaces.Discrete(nrow * ncol + 3)
            self.observation_space.shape = (nrow * ncol + 3, )

        self._reset()

    def _random_map(self):
        return np.asarray(random.choice(list(MAPS.values())), dtype='c')


    def _reset(self):
        """
        direction N, W, S, E = 0, 1, 2, 3
        => player will be 5, 6, 7, 8
        """
        if self.use_random_map:
            self.desc = desc = self._random_map()
            self.map, self.start_pos, self.goal_pos = self._build_map()
            self.built_done = True
        elif self.built_done is False:
            self.map, self.start_pos, self.goal_pos = self._build_map()
            self._map = self.map.copy()
            self.built_done = True
        else :
            self.map = self._map.copy()

        self.step_cnt             = 0
        self._orig_block          = 0
        self.player_pos           = self.start_pos
        self.player_dir           = np.random.randint(4)
        self.cur_desc             = self.desc.copy()

        if self.put_player:
            self.map[self.player_pos] = PLAYER + self.player_dir

        return self._get_observation()[0]

    def _build_map(self):

        m = np.zeros((self.nrow, self.ncol), dtype='int')
        sp = (0, 0)
        gp = (0, 0)

        for r in range(self.nrow):
            for c in range(self.ncol):
                if   self.desc[r, c] == b'S' :
                    sp = r, c
                elif self.desc[r, c] == b'B' :
                    m[r, c] = BRIDGE
                elif self.desc[r, c] == b'R' :
                    m[r, c] = RIVER
                elif self.desc[r, c] == b'G' :
                    m[r, c] = GOAL
                    gp = r, c

        return m, sp, gp

    def _step(self, a):

        if   a == MOVE        : self.player_pos = self._move_forward()
        elif a == TURN_LEFT   : self.player_dir = self._turn_dir(1)
        elif a == TURN_RIGHT  : self.player_dir = self._turn_dir(-1)
        elif a == PLACE_BLOCK : self._place_block()

        self.step_cnt   += 1
        self.total_step += 1
        self._pa = a

        return self._get_observation()

    def _turn_dir(self, direction):
        r, c            = self.player_pos
        new_dir         = (self.player_dir + direction) % 4
        if self.put_player :
            self.map[r, c]  = PLAYER + new_dir

        return new_dir

    def _move_forward(self):
        r, c                   = self._get_front(self.player_pos, self.player_dir)
        old_r, old_c           = self.player_pos
        self.map[old_r, old_c] = self._orig_block
        self._orig_block       = self.map[r, c]
        if self.put_player :
            self.map[r, c] = PLAYER + self.player_dir

        return r, c

    def _place_block(self):

        r, c = self._get_front(self.player_pos, self.player_dir)
        if self.map[r, c] == RIVER :
            self.map[r, c] = BRIDGE
            self.cur_desc[r, c] = 'B'

    def _get_observation(self):

        state = self.map if self.use_flatten is False else self.map.flatten()
        if self.give_details: state = np.append(state, [self.player_pos[0], self.player_pos[1], self.player_dir])

        if self.player_pos == self.goal_pos :
            done   = True
            reward = REWARD
        elif self.step_cnt > self.time_limit :
            done   = True
            reward = TIMEOUT
        elif self._orig_block == RIVER:
            self.map[self.player_pos] = DROWN
            done   = True
            reward = DIE
        else :
            done   = False
            reward = self.step_penalty

        if self.step_render :
            self._render()
            if hasattr(self, '_pa'):
                if reward > self._best:
                    self._best = reward
                print("reward : {}({}), done : {}, action : {}".format(reward, self._best, done, ['MOVE', 'TURN_LEFT', 'TURN_RIGHT', 'PLACE'][self._pa]))

        return (state, reward, done, None)

    def _get_partial_observation(self):
        pass


    def _get_front(self, pos, direction):
        r, c = pos

        if   direction == 0: r = np.maximum(r - 1, 0)
        elif direction == 1: c = np.maximum(c - 1, 0)
        elif direction == 2: r = np.minimum(r + 1, self.nrow - 1)
        elif direction == 3: c = np.minimum(c + 1, self.ncol - 1)

        return r, c

    def _render(self, mode='human', close=False):
        if close : return

        row, col = self.player_pos
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.cur_desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        outfile.write("\nDirection : {}, Action Count : {}({})\n"
                .format(["UP", "LEFT", "DOWN", "RIGHT"][self.player_dir], self.step_cnt, self.total_step))
        outfile.write("\n".join(''.join(line) for line in desc)+'\n')

        if mode != 'human' : return outfile

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

if __name__ == '__main__':

    BBE = BuildBridgeEnv(use_random_map=True, use_time_limit=True, put_player=True, step_render=True)
    while True :
        #BBE.render()
        key = input()
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

        if d : BBE.reset()

