from curses import wrapper

from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import wrappers

def make_env():
    #env = simple_tag_v3.env(num_obstacles=0)
    env = simple_tag_v3.env()
    #env = wrappers.OrderEnforcingWrapper(env)
    return env