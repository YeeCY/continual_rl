from dm_control import composer
from dm_control import viewer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.tasks import go_to_target
from dm_control.locomotion.tasks import escape
from dm_control.locomotion.examples import basic_cmu_2019, basic_rodent_2020
from dm_control.locomotion.walkers import ant, jumping_ball, rodent, initializers
from dm_control.locomotion.walkers.planar_walker import PlanarWalker


_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.005


def ant_run(random_state=None):
    walker = ant.Ant()
    arena = corr_arenas.EmptyCorridor()
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(5, 0, 0),
        walker_spawn_rotation=0,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def ant_run_long(random_state=None):
    walker = ant.Ant()
    arena = corr_arenas.EmptyCorridor(
        corridor_length=250,
        visible_side_planes=False)
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1, 0, 0),
        walker_spawn_rotation=0,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def ant_run_walls(random_state=None):
    walker = ant.Ant()

    arena = corr_arenas.WallsCorridor(
        wall_gap=4.,
        wall_width=distributions.Uniform(1, 7),
        wall_height=3.0,
        corridor_width=10,
        corridor_length=100,
        include_initial_padding=False)

    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        walker_spawn_rotation=0,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def ant_run_gaps(random_state=None):
    walker = ant.Ant()

    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(1.25, 2.5),  # (0.3, 2.5)
        gap_length=distributions.Uniform(0.3, 0.7),  # (0.5, 1.25)
        corridor_width=10,
        corridor_length=250)

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1.0, 0, 0),
        target_velocity=3.0,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def ant_escape_bowl(random_state=None):
    walker = ant.Ant()

    # Build a bowl-shaped arena.
    arena = bowl.Bowl(ground_size=(15., 15.), hfield_size=(30, 30, 5), terrain_smoothness=0.15, terrain_bump_scale=2.0)

    # Build a task that rewards the agent for being far from the origin.
    task = escape.Escape(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0, 0, 1.5),
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(time_limit=30,  # 20
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def rolling_ball_with_head_run(random_state=None):
    walker = jumping_ball.RollingBallWithHead()
    arena = corr_arenas.EmptyCorridor()
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(5, 0, 0),
        walker_spawn_rotation=0,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def jumping_ball_run(random_state=None):
    walker = jumping_ball.JumpingBallWithHead()
    arena = corr_arenas.EmptyCorridor()
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(5, 0, 0),
        walker_spawn_rotation=0,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def jumping_ball_run_gaps(random_state=None):
    walker = jumping_ball.JumpingBallWithHead()

    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(1.0, 2.5),  # (0.3, 2.5)
        gap_length=distributions.Uniform(0.3, 0.7),  # (0.5, 1.25)
        corridor_width=10,
        corridor_length=250)

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1.0, 0, 0),
        target_velocity=3.0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(time_limit=30,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def jumping_ball_go_to_target(random_state=None):
    walker = jumping_ball.JumpingBallWithHead()

    # Build a standard floor arena.
    arena = floors.Floor()

    # Build a task that rewards the agent for going to a target.
    task = go_to_target.GoToTarget(
        walker=walker,
        arena=arena,
        sparse_reward=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(time_limit=30,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def walker_run():
    walker = PlanarWalker(initializer=initializers.RandomJointPositionInitializer())
    arena = corr_arenas.LongCorridor()
    task = corr_tasks.PlanarRunThroughCorridor(
        walker=walker,
        arena=arena,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def walker_run_long():
    walker = PlanarWalker()
    arena = corr_arenas.EmptyCorridor(
        corridor_length=250,
        visible_side_planes=False)
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1, 0, 0),
        walker_spawn_rotation=0,
        stand_height=1.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def walker_run_gaps(random_state=None):
    walker = PlanarWalker()

    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(1.25, 2.5),  # (0.3, 2.5)
        gap_length=distributions.Uniform(0.3, 0.6),  # (0.5, 1.25)
        corridor_width=10,
        corridor_length=250)

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1.0, 0, 0),
        stand_height=1.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(time_limit=30,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def main():
    # viewer.launch(environment_loader=ant_run)
    # viewer.launch(environment_loader=ant_run_long)
    # viewer.launch(environment_loader=ant_run_walls)
    viewer.launch(environment_loader=ant_run_gaps)
    # viewer.launch(environment_loader=ant_escape_bowl)
    # viewer.launch(environment_loader=rolling_ball_with_head_run)
    # viewer.launch(environment_loader=jumping_ball_run)
    # viewer.launch(environment_loader=jumping_ball_run_gaps)
    # viewer.launch(environment_loader=jumping_ball_go_to_target)
    # viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_walls)
    # viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_gaps)
    # viewer.launch(environment_loader=walker_run)
    # viewer.launch(environment_loader=walker_run_long)
    # viewer.launch(environment_loader=walker_run_gaps)

    # # Build an example environment.
    # import numpy as np
    # import dmc2gym
    # import gym
    #
    # # observation_shape:
    # #   walker_run = 19 without range finders, 28 with range finders
    # #   ant_run_long = 26 without range finders, 37 with range finders
    # #   jumping_ball_run_long = 10 without range finder, 19 with range finder
    # # action_shape:
    # #   walker_run_long =
    # #   walker_run_gaps
    # #   ant_run_long
    # environment = dmc2gym.make_locomotion(
    #     env_name='walker_run_gaps',
    #     seed=0,
    #     from_pixels=False,
    #     episode_length=1000,
    # )
    #
    # gym_env = gym.make('Ant-v3')
    # observation_spec = gym_env.observation_space
    # obs = gym_env.reset()
    # obs, reward, done, info = gym_env.step(gym_env.action_space.sample())
    #
    # # observation_shape:
    # #   walker_run = 24
    # # suite_env = dmc2gym.make(
    # #     domain_name='walker',
    # #     task_name='run',
    # #     seed=0,
    # #     from_pixels=False,
    # #     episode_length=1000,
    # # )
    #
    # # Get the `action_spec` describing the control inputs.
    # action_spec = environment.action_spec()
    #
    # # Step through the environment for one episode with random actions.
    # done = False
    # environment.reset()
    # while not done:
    #     action = np.random.uniform(action_spec.minimum, action_spec.maximum,
    #                                size=action_spec.shape)
    #     obs, reward, done, info = environment.step(action)
    #     print("obs = {}, reward = {}, done = {}, info = {}".format(obs, reward, done, info))
    #
    # print("done")


if __name__ == "__main__":
    main()
