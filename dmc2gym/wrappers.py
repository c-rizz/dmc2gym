from gym import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np
import gym


def spec_to_gym(spec):
    if type(spec) == specs.BoundedArray:
        low = spec.minimum.item() if np.ndim(spec.minimum)==0 else spec.minimum
        high = spec.maximum.item() if np.ndim(spec.maximum)==0 else spec.maximum
        # print(f"low = {low} scalar = {np.isscalar(low)} shape = {spec.shape}")
        space = gym.spaces.Box( low = low,
                                high = high,
                                shape = spec.shape,
                                dtype = spec.dtype)
    elif type(spec) == specs.Array:
        space = gym.spaces.Box( low = spec.dtype("-inf"),
                                high = spec.dtype("+inf"),
                                shape = spec.shape,
                                dtype = spec.dtype)
    elif type(spec) == dict:
        spacedict = {}
        for subspec_name, subspec in spec.items():
            spacedict[subspec_name] = spec_to_gym(subspec)
        space = gym.spaces.Dict(spacedict)
    else:
        raise AttributeError(f"Unsupported spec type "+str(type(spec)))
    return space
            

def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)




class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name = None,
        task_name = None,
        env : core.Env = None,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        if env is not None:
            if domain_name is not None or task_name is not None:
                raise AttributeError(f"env is not None but domain_name = {domain_name} and task_name = {task_name}. Can only use either env or (domain,task).")
            self._env = env
        elif domain_name is not None and task_name is not None:
            # create task
            self._env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs
            )
        else:
            raise AttributeError(f"No valid env selected: domain_name = {domain_name}, task_name = {task_name}, env = {env}")

        # true and normalized action spaces
        self._true_action_space = spec_to_gym(self._env.action_spec()) #_spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = spec_to_gym(self._env.observation_spec())
            
        self._state_space = spec_to_gym(self._env.action_spec())
        
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = time_step.observation
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = time_step.observation
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = time_step.observation
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
