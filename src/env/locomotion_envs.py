from dm_control import composer
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import ant, planar_walker, initializers


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
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)
