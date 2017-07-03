import gym, sys
from build_bridge import BuildBridgeEnv
from baselines import deepq

START_LINE = "SLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"
LAND_LINE  = "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"
GOAL_LINE  = "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLGLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"
RIVER_LINE = "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR"

def generate_desc(num_of_river=10):
    cnt = 0
    a = [START_LINE]

    for x in range(1, 99):
        if x > 50 and cnt < num_of_river:
            cnt += 1
            a.append(RIVER_LINE)
        else :
            a.append(LAND_LINE)
    a.append(GOAL_LINE)
    return a

def main(args):
    desc = generate_desc()
    env   = BuildBridgeEnv(map_name=args.map_name, use_random_map=args.use_random,
            use_time_limit=True, put_player=not(args.simple), use_partial=args.use_partial,
            give_details=args.use_details, step_penalty=-0.001, time_limit=500,
            use_flatten=True, step_render=True, extra_dim=False, use_delay_reset=args.use_delay,
            use_coord=args.use_coord, simple_action=args.simple)
    model = deepq.models.mlp([256, 256, 128, 128, 64, 64, 8])

    act   = deepq.learn(
                env,
                q_func=model,
                lr=5e-4,
                print_freq=20,
                exploration_fraction=0.1,
                exploration_final_eps=0.1,
                gamma=0.99,
                max_timesteps=1000000,
                prioritized_replay=args.prioritized,
                callback=None
            )
    if args.mdl_name is not None :
        act.save(args.mdl_name)

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description='parsing training args')
    parser.add_argument('--simple', dest='simple', action='store_true')
    parser.add_argument('--coord', dest='use_coord', action='store_true')
    parser.add_argument('--random', dest='use_random', action='store_true')
    parser.add_argument('--partial', dest='use_partial', action='store_true')
    parser.add_argument('--details', dest='use_details', action='store_true')
    parser.add_argument('--fname', dest='mdl_name', type=str)
    parser.add_argument('--prioritized', dest='prioritized', action='store_true')
    parser.add_argument('--delay', dest='use_delay', action='store_true')
    parser.add_argument('--map_name', dest='map_name', type=str)
    args = parser.parse_args()
    main(args)

