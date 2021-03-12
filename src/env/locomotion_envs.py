from dm_control import composer
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import ant, jumping_ball, planar_walker, initializers
from dm_control.composer.variation import distributions


_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.005


def walker_run():
    walker = planar_walker.PlanarWalker(initializer=initializers.RandomJointPositionInitializer())
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


def ant_run():
    walker = ant.Ant()
    arena = corr_arenas.EmptyCorridor()
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(5, 0, 0),
        walker_spawn_rotation=0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def ant_run_long():
    walker = ant.Ant()
    arena = corr_arenas.EmptyCorridor(
        corridor_length=250,
        visible_side_planes=False)
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1, 0, 0),
        walker_spawn_rotation=0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def ant_run_walls():
    walker = ant.Ant()

    arena = corr_arenas.WallsCorridor(
        wall_gap=4.,
        wall_width=distributions.Uniform(1, 7),
        wall_height=3.0,
        corridor_width=10,
        corridor_length=250,
        include_initial_padding=False)

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
        strip_singleton_obs_buffer_dim=True)


def ant_run_gaps():
    walker = ant.Ant()

    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(1.25, 2.5),  # (0.3, 2.5)
        gap_length=distributions.Uniform(0.3, 0.8),  # (0.5, 1.25)
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


def jumping_ball_run_long():
    walker = jumping_ball.JumpingBallWithHead()

    arena = corr_arenas.EmptyCorridor(
        corridor_length=250,
        visible_side_planes=False)
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(1, 0, 0),
        walker_spawn_rotation=0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def jumping_ball_run_walls():
    walker = jumping_ball.JumpingBallWithHead()

    arena = corr_arenas.WallsCorridor(
        wall_gap=4.,
        wall_width=distributions.Uniform(1, 7),
        wall_height=3.0,
        corridor_width=10,
        corridor_length=250,
        include_initial_padding=False)

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
