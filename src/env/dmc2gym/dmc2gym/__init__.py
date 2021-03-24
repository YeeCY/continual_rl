import gym
from gym.envs.registration import register
from src.env import locomotion_envs


def make(
        domain_name,
        task_name,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        setting_kwargs=None,
        time_limit=1e6,
        channels_first=True
):
    env_id = 'dmc_%s_%s-v1' % (domain_name, task_name)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='dmc2gym.wrappers:DMCSuiteWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                setting_kwargs=setting_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)


def make_locomotion(
    env_name,
    seed=1,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    time_limit=1e6,
    channels_first=True
):
    assert hasattr(locomotion_envs, env_name), "please use valid locomotion environments"
    env_id = 'dmc_loco_%s-v1' % (env_name)

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='dmc2gym.wrappers:DMCLocomotionWrapper',
            kwargs=dict(
                env_name=env_name,
                task_kwargs=task_kwargs,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)

