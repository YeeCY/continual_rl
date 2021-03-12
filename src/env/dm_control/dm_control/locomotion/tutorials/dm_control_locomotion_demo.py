from dm_control import composer
from dm_control import viewer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.examples import basic_cmu_2019
from dm_control.locomotion.walkers import ant, jumping_ball, initializers
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
        platform_length=distributions.Uniform(.3, 2.5),
        gap_length=distributions.Uniform(.5, 1.0),
        corridor_width=10,
        corridor_length=100)

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


def jumping_ball_run_gaps():
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

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def walker_run(random_state=None):
    # walker = PlanarWalker(initializer=initializers.RandomJointPositionInitializer())
    walker = PlanarWalker()
    arena = corr_arenas.EmptyCorridor()
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        walker_spawn_rotation=0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def walker_run_long(random_state=None):
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
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)


def walker_run_gaps(random_state=None):
    walker = PlanarWalker(initializer=initializers.RandomJointPositionInitializer())

    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(.3, 2.5),
        gap_length=distributions.Uniform(.5, 1.25),
        corridor_width=10,
        corridor_length=100)

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


def main():
    # viewer.launch(environment_loader=ant_run)
    # viewer.launch(ant_run_long)
    # viewer.launch(environment_loader=ant_run_walls)
    # viewer.launch(environment_loader=ant_run_gaps)
    # viewer.launch(environment_loader=rolling_ball_with_head_run)
    # viewer.launch(environment_loader=jumping_ball_run)
    viewer.launch(environment_loader=jumping_ball_run_gaps)
    # viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_walls)
    # viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_gaps)
    # viewer.launch(environment_loader=walker_run)
    # viewer.launch(environment_loader=walker_run_gaps)

    # # Build an example environment.
    # import numpy as np
    #
    # env = ant_run_long()
    #
    # # Get the `action_spec` describing the control inputs.
    # action_spec = env.action_spec()
    #
    # # Step through the environment for one episode with random actions.
    # time_step = env.reset()
    # while not time_step.last():
    #     action = np.random.uniform(action_spec.minimum, action_spec.maximum,
    #                                size=action_spec.shape)
    #     time_step = env.step(action)
    #     print("reward = {}, discount = {}, observations = {}.".format(
    #         time_step.reward, time_step.discount, time_step.observation))
    #
    # print("done")


if __name__ == "__main__":
    main()
