from dm_control import composer
from dm_control import viewer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import ant, jumping_ball
from dm_control.locomotion.walkers.walker import Walker

from dm_control.locomotion.examples import basic_cmu_2019

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.005


def ant_run(randon_state=None):
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
        random_state=randon_state,
        strip_singleton_obs_buffer_dim=True)


def ant_run_walls(randon_state=None):
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
        random_state=randon_state,
        strip_singleton_obs_buffer_dim=True)


def rolling_ball_with_head_run(randon_state=None):
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
        random_state=randon_state,
        strip_singleton_obs_buffer_dim=True)


def jumping_ball_with_head_run(randon_state=None):
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
        random_state=randon_state,
        strip_singleton_obs_buffer_dim=True)


def walker_run(randon_state=None):
    walker = Walker()
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
        random_state=randon_state,
        strip_singleton_obs_buffer_dim=True)

def main():
    # viewer.launch(environment_loader=ant_run)
    # viewer.launch(environment_loader=ant_run_walls)
    # viewer.launch(environment_loader=rolling_ball_with_head_run)
    # viewer.launch(environment_loader=jumping_ball_with_head_run)
    # viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_walls)
    # viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_gaps)
    viewer.launch(environment_loader=walker_run)


if __name__ == "__main__":
    main()
