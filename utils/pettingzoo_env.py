from pettingzoo.mpe import simple_tag_v3
#from pettingzoo.utils import wrappers

def make_env():
    env = simple_tag_v3.raw_env()
    #env = wrappers.OrderEnforcingWrapper(env)
    return env