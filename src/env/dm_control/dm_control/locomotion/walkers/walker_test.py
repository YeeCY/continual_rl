"""Tests for the Walker."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers.walker import Walker
import numpy as np

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.005


def _get_ant_corridor_physics():
  walker = Walker()
  arena = corr_arenas.EmptyCorridor()
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(5, 0, 0),
      walker_spawn_rotation=0,
      physics_timestep=_PHYSICS_TIMESTEP,
      control_timestep=_CONTROL_TIMESTEP)

  env = composer.Environment(
      time_limit=30,
      task=task,
      strip_singleton_obs_buffer_dim=True)

  return walker, env


class WalkerTest(parameterized.TestCase):

  def test_can_compile_and_step_simulation(self):
    _, env = _get_ant_corridor_physics()
    physics = env.physics
    for _ in range(100):
      physics.step()

  @parameterized.parameters([
      'egocentric_camera',
      'root_body',
  ])
  def test_get_element_property(self, name):
    attribute_value = getattr(Walker(), name)
    self.assertIsInstance(attribute_value, mjcf.Element)

  @parameterized.parameters([
      'actuators',
      'end_effectors',
      'observable_joints',
  ])
  def test_get_element_tuple_property(self, name):
    attribute_value = getattr(Walker(), name)
    self.assertNotEmpty(attribute_value)
    for item in attribute_value:
      self.assertIsInstance(item, mjcf.Element)

  def test_set_name(self):
    name = 'fred'
    walker = Walker(name=name)
    self.assertEqual(walker.mjcf_model.model, name)

  @parameterized.parameters(
      'appendages_pos',
      'sensors_touch',
  )
  def test_evaluate_observable(self, name):
    walker, env = _get_ant_corridor_physics()
    physics = env.physics
    observable = getattr(walker.observables, name)
    observation = observable(physics)
    self.assertIsInstance(observation, (float, np.ndarray))

  def test_proprioception(self):
    walker = Walker()
    for item in walker.observables.proprioception:
      self.assertIsInstance(item, observable_base.Observable)


if __name__ == '__main__':
  absltest.main()
