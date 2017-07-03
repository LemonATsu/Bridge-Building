import gym, sys, pickle, time
import argparse
from build_bridge import BuildBridgeEnv
from baselines import deepq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parsing training args')
    parser.add_argument('--large', dest='use_large', action='store_true')
    parser.add_argument('--simple', dest='simple', action='store_true')
    parser.add_argument('--random', dest='use_random', action='store_true')
    parser.add_argument('--partial', dest='use_partial', action='store_true')
    parser.add_argument('--details', dest='use_details', action='store_true')
    parser.add_argument('--mdl', dest='mdl', type=str)
    parser.add_argument('--prioritized', dest='prioritized', action='store_true')
    parser.add_argument('--delay', dest='use_delay', action='store_true')
    parser.add_argument('--map_name', dest='map_name', type=str)

    args = parser.parse_args()
    env  = BuildBridgeEnv(map_name=args.map_name, use_random_map=args.use_random,
                use_time_limit=True, put_player=not(args.simple), use_partial=args.use_partial,
                give_details=args.use_details, step_penalty=-0.001, time_limit=500,
                use_flatten=True, step_render=True, extra_dim=False, use_delay_reset=args.use_delay, simple_action=args.simple)


    act = deepq.load(args.mdl)

    eval_episode = 50
    eval_cnt = 0
    acc_reward = 0.

    while eval_cnt < eval_episode:
        eval_cnt += 1
        obs, d = env.reset(), False
        episode_r = 0
        while not d :
            obs, r, d, _ = env.step(act(obs[None])[0])
            episode_r += r
            time.sleep(0.01)
        acc_reward += episode_r

    print("Mean reward : {}".format(acc_reward / eval_episode))


