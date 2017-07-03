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


BRIDGE      = 1
GOAL        = 2
PLAYER      = 3
RIVER       = 7
DROWN       = 8
BLOCKED     = 9

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


class BuildBridgeEnv(Env):
    """
    S : starting point
    L : land
    B : bridge
    R : river, cannot pass it unless you have built bridge
    G : goal
    """

    metadata = {'render.modes' : ['human', 'ansi']}


    def __init__(self, desc=None, map_name=None, step_penalty=0, use_random_map=False, use_coord=False,
                 give_details=False, put_player=True, use_delay_reset=False, delay_reset=5, delay_reset_times=500,
                 use_time_limit=False, time_limit=40, use_flatten=False, step_render=False, simple_action=False,
                 extra_dim=True, use_partial=False, partial_dist=3):
        """
        desc             : map description, as shown in MAPS var.
        map_name         : predefined map
        step_penalty     : penalty received after each step you have take
        use_random_map   : randomly select map from MAPS
        use_coord        : only return the agent's coordinates as observation
        give_details     : return the exact position/direction of player.
                           use_flatten must be set to True when using this.
        put_player       : put player on the returned state(map)
        use_delay_reset  : only reset after n episodes
        delay_reset      : number of episodes before the environment reset
        delay_reset_times: number of resets before the delay reset mechanism is disabled
        use_time_limit   : limit the number of step(time) the player can take
        time_limit       : the amount of step that a player can take when
                           use_time_limit is set to True
        step_render      : render after each step
        extra_dim        : extend dimension from (a, b) to (a, b, 1)
        use_parital      : partial observation
        partial_dist     : the distance of state that you can observe in partial observation mode
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
        self.use_flatten       = use_flatten
        self.give_details      = give_details
        self.put_player        = put_player
        self.total_step        = 0
        self._best             = -100
        self.step_render       = step_render
        self.extra_dim         = extra_dim
        self.use_partial       = use_partial
        self.partial_dist      = partial_dist
        self.use_delay_reset   = use_delay_reset
        self.delay_reset       = delay_reset
        self.delay_reset_times = delay_reset_times
        self.delay_cnt         = 0
        self.delay_reset_cnt   = 0
        self.use_coord         = use_coord
        self.simple_action     = simple_action
        self.action_space      = spaces.Discrete(4) if not self.simple_action else spaces.Discrete(8)

        if use_coord :
            self.observation_space = spaces.Discrete(2)
            self.observation_space.shape = (2,)
        elif give_details is False and use_partial is False:
            self.observation_space = spaces.Discrete(nrow * ncol)
            self.observation_space.shape = (nrow, ncol) if use_flatten is False else (nrow * ncol, )
        elif use_partial :
            state_len = partial_dist if give_details is False else partial_dist + 3
            self.observation_space = spaces.Discrete(state_len)
            self.observation_space.shape = (state_len, )
        else :
            state_len = nrow * ncol + 3
            self.observation_space = spaces.Discrete(state_len)
            self.observation_space.shape = (state_len, )


        self._reset()

    def _random_map(self):
        return np.asarray(random.choice(list(MAPS.values())), dtype='c')


    def _reset(self):
        """
        direction N, W, S, E = 0, 1, 2, 3
        => player will be 5, 6, 7, 8
        """
        if self.use_delay_reset and self.built_done and self.step_cnt < self.time_limit:

            self.delay_cnt += 1
            if self.delay_cnt % self.delay_reset != 0:
                self.step_cnt = 0
                r, c = self.player_pos
                self.map[r, c] = self._orig_block
                self.player_pos  = self.start_pos

                if self.put_player :
                    self._put_player()

                self._orig_block = 0
                return self._get_observation()[0]
            else :
                self.delay_cnt = 0
                self.delay_reset_cnt += 1
                if self.delay_reset_cnt == self.delay_reset_times : self.use_delay_reset = False

        if self.use_random_map :
            self.desc = desc = self._random_map()
            self.map, self.start_pos, self.goal_pos = self._build_map()
            self.built_done = True
        elif self.built_done is False :
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
            self._put_player()

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

        """
        if   a == MOVE        : self.player_pos = self._move_forward()
        elif a == TURN_LEFT   : self.player_dir = self._turn_dir(1)
        elif a == TURN_RIGHT  : self.player_dir = self._turn_dir(-1)
        elif a == PLACE_BLOCK : self._place_block()
        """

        if self.simple_action : self._step_simple(a)
        else : self._step_hard(a)

        if self.put_player :
            self._put_player()

        self.step_cnt   += 1
        self.total_step += 1
        self._pa = a

        return self._get_observation()

    def _step_hard(self, a):
        if   a == MOVE        : self.player_pos = self._move_forward()
        elif a == TURN_LEFT   : self.player_dir = self._turn_dir(1)
        elif a == TURN_RIGHT  : self.player_dir = self._turn_dir(-1)
        elif a == PLACE_BLOCK : self._place_block()

    def _step_simple(self, a):
        if   a == MOVE_UP   :
            self.player_dir = UP
            self.player_pos = self._move_forward()
        elif a == MOVE_LEFT :
            self.player_dir = LEFT
            self.player_pos = self._move_forward()
        elif a == MOVE_DOWN :
            self.player_dir = DOWN
            self.player_pos = self._move_forward()
        elif a == MOVE_RIGHT:
            self.player_dir = RIGHT
            self.player_pos = self._move_forward()
        elif a == PLACE_UP :
            self.player_dir = UP
            self._place_block()
        elif a == PLACE_LEFT :
            self.player_dir = LEFT
            self._place_block()
        elif a == PLACE_DOWN :
            self.player_dir = DOWN
            self._place_block()
        elif a == PLACE_RIGHT :
            self.player_dir = RIGHT
            self._place_block()

    def _put_player(self):

        self.map[self.player_pos] = PLAYER
        if self.simple_action :
            self.map[self.player_pos] += self.player_dir


    def _turn_dir(self, direction):
        return (self.player_dir + direction) % 4

    def _move_forward(self):
        r, c                   = self._get_front(self.player_pos, self.player_dir)
        old_r, old_c           = self.player_pos
        self.map[old_r, old_c] = self._orig_block
        self._orig_block       = self.map[r, c]

        return r, c

    def _place_block(self):

        r, c = self._get_front(self.player_pos, self.player_dir)
        if self.map[r, c] == RIVER :
            self.map[r, c] = BRIDGE
            self.cur_desc[r, c] = 'B'

    def _get_observation(self):

        if self.use_coord:
            state = np.array(self.player_pos)
        elif self.use_partial :
            state = self._get_partial_state()
        else :
            state = self.map if self.use_flatten is not True else self.map.flatten()

        if self.give_details:
            state = np.append(state, [self.player_pos[0], self.player_pos[1], self.player_dir])

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
                if self.simple_action : action_name = ['UP', 'LEFT', 'DOWN', 'RIGHT', 'P_UP', 'P_LEFT', 'P_DOWN', 'P_RIGHT']
                else : action_name = ['MOVE', 'TURN_LEFT', 'TURN_RIGHT', 'PLACE']
                print("reward : {}({}), done : {}, action : {}, position : {}".format(
                    reward, self._best, done, action_name[self._pa], state))


        if self.extra_dim is True :
            if not self.use_flatten and not self.use_partial and not self.use_coord:
                state = state.reshape(self.nrow, self.ncol, 1)
        return (state, reward, done, None)

    def _get_partial_state(self):
        state = np.array([9] * (self.partial_dist))

        if self._at_boundary() : return state

        r, c = self._get_front(self.player_pos, self.player_dir)
        r_end, c_end = self._get_dir_dist(self.player_pos, self.player_dir, self.partial_dist)

        flip = False
        if r > r_end or c > c_end:
            r, r_end = r_end, r
            c, c_end = c_end, c
            flip = True

        e_len = np.maximum(r_end - r + 1, c_end - c + 1)
        state[:e_len] = self.map[r:r_end+1,c:c_end+1].flatten()
        if flip : state = np.flipud(state)

        return state

    def _get_front(self, pos, direction):
        return self._get_dir_dist(pos, direction, 1)

    def _get_dir_dist(self, pos, direction, dist):
        r, c = pos

        if   direction == 0: r = np.maximum(r - dist, 0)
        elif direction == 1: c = np.maximum(c - dist, 0)
        elif direction == 2: r = np.minimum(r + dist, self.nrow - 1)
        elif direction == 3: c = np.minimum(c + dist, self.ncol - 1)

        return r, c

    def _at_boundary(self):
        r, c = self.player_pos
        result = False
        if (r == 0 and self.player_dir == 0) : result = True
        if (c == 0 and self.player_dir == 1) : result = True
        if (r == self.nrow - 1 and self.player_dir == 2) : result = True
        if (c == self.ncol - 1 and self.player_dir == 3) : result = True

        return result



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

    BBE = BuildBridgeEnv(map_name='8x8', use_coord=True,  step_render=True, extra_dim=False)
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

        if d :
            print("done!")
            BBE.reset()

