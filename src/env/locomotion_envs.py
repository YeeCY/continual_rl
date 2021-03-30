from dm_control import composer
from dm_control.composer.variation import distributions

from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.tasks import go_to_target
from dm_control.locomotion.tasks import escape
from dm_control.locomotion.walkers import ant, jumping_ball, planar_walker, initializers

from dm_control.utils import rewards


_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.005


def _walker_get_reward(self, physics):
    walker_height = physics.bind(self._walker.root_body).xpos[2]  # xpos['z']
    stand_reward = rewards.tolerance(walker_height,
                                     bounds=(self._height, float('inf')),
                                     margin=self._height / 2)

    walker_vel = physics.bind(self._walker.root_body).subtree_linvel[0]
    move_reward = rewards.tolerance(walker_vel,
                                    bounds=(self._vel, float('inf')),
                                    margin=self._vel / 2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
    return stand_reward * (5 * move_reward + 1) / 6


def _ant_get_reward(self, physics):
    walker_height = physics.bind(self._walker.root_body).xpos[2]  # xpos['z']
    standing = rewards.tolerance(walker_height,
                                 bounds=(self._height, float('inf')),
                                 margin=self._height / 2)
    walker_upright = physics.bind(self._walker.root_body).xmat[-1]  # xmat['zz']

    upright = (1 + walker_upright) / 2
    stand_reward = (3 * standing + upright) / 4

    walker_vel = physics.bind(self._walker.root_body).subtree_linvel[0]
    move_reward = rewards.tolerance(walker_vel,
                                    bounds=(self._vel, float('inf')),
                                    margin=self._vel / 2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
    return stand_reward * (5 * move_reward + 1) / 6


def walker_run():
    walker = planar_walker.PlanarWalker(initializer=initializers.RandomJointPositionInitializer())
    arena = corr_arenas.LongCorridor()
    task = corr_tasks.PlanarRunThroughCorridor(
        walker=walker,
        arena=arena,
        stand_height=1.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    # (Chongyi Zheng): redefine reward function
    #   https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
    task.get_reward = _walker_get_reward.__get__(task, task.get_reward)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def walker_run_long():
    walker = planar_walker.PlanarWalker()
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

    # (Chongyi Zheng): redefine reward function
    task.get_reward = _walker_get_reward.__get__(task, task.get_reward)

    return composer.Environment(
        time_limit=30,
        task=task,
        strip_singleton_obs_buffer_dim=True)


def walker_run_gaps(random_state=None):
    walker = planar_walker.PlanarWalker()

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
        stand_height=1.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    # (Chongyi Zheng): redefine reward function
    task.get_reward = _walker_get_reward.__get__(task, task.get_reward)

    return composer.Environment(time_limit=30,
                                task=task,
                                random_state=random_state,
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
        stand_height=0.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    # (Chongyi Zheng): redefine reward function
    # task.get_reward = _ant_get_reward.__get__(task, task.get_reward)

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
        stand_height=0.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    # (Chongyi Zheng): redefine reward function
    # task.get_reward = _ant_get_reward.__get__(task, task.get_reward)

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
        stand_height=0.2,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    # (Chongyi Zheng): redefine reward function
    # task.get_reward = _ant_get_reward.__get__(task, task.get_reward)

    return composer.Environment(
        time_limit=30,
        task=task,
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
        platform_length=distributions.Uniform(0.3, 2.5),  # (0.3, 2.5)
        gap_length=distributions.Uniform(0.5, 1.25),  # (0.5, 1.25)
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
