"""A two leg "walker" walker."""

import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import math as mjmath
import numpy as np


_XML_PATH = os.path.join(os.path.dirname(__file__),
                            'assets/walker.xml')


class PlanarWalker(legacy_base.Walker):
    """A two leg "PlanarWalker" walker."""

    def _build(self, name='walker', marker_rgba=None, initializer=None):
        """Build an PlanarWalker walker.

        Args:
            name: name of the walker.
            marker_rgba: (Optional) color the walker's legs with marker_rgba.
            initializer: (Optional) A `WalkerInitializer` object.
        """
        super()._build(initializer=initializer)
        self._mjcf_root = mjcf.from_path(_XML_PATH)
        if name:
            self._mjcf_root.model = name

        # Set corresponding marker color if specified.
        if marker_rgba is not None:
            for geom in self.marker_geoms:
                geom.set_attributes(rgba=marker_rgba)

        # Initialize previous action.
        self._prev_action = np.zeros(shape=self.action_spec.shape,
                                     dtype=self.action_spec.dtype)

    def create_root_joints(self, attachment_frame):
        ### Planar attachment joint
        attachment_frame.add('joint', name='root_z', axis='0 0 1', type='slide', limited='false',
                             armature='0', damping='0')
        attachment_frame.add('joint', name='root_x', axis='1 0 0', type='slide', limited='false',
                             armature='0', damping='0')
        attachment_frame.add('joint', name='root_y', axis='0 1 0', type='hinge', limited='false',
                             armature='0', damping='0')  # can only rotate around y axis

    def initialize_episode(self, physics, random_state):
        self._prev_action = np.zeros(shape=self.action_spec.shape,
                                     dtype=self.action_spec.dtype)

    def apply_action(self, physics, action, random_state):
        super().apply_action(physics, action, random_state)

        # Updates previous action.
        self._prev_action[:] = action

    def _build_observables(self):
        return PlanarWalkerObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def marker_geoms(self):
        return [self._mjcf_root.find('geom', 'left_leg'),
                self._mjcf_root.find('geom', 'right_leg')]

    @composer.cached_property
    def actuators(self):
        return self._mjcf_root.find_all('actuator')

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'torso')

    @composer.cached_property
    def bodies(self):
        return tuple(self.mjcf_model.find_all('body'))

    @property
    def _foot_bodies(self):
        return (self._mjcf_root.find('body', 'left_foot'),
                self._mjcf_root.find('body', 'right_foot'))

    @composer.cached_property
    def end_effectors(self):
        return self._foot_bodies

    @composer.cached_property
    def observable_joints(self):
        return tuple(actuator.joint for actuator in self.actuators)

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    # def aliveness(self, physics):
    #     return (physics.bind(self.root_body).xmat[-1] - 1.) / 2.

    @composer.cached_property
    def ground_contact_geoms(self):
        foot_geoms = []
        for foot in self._foot_bodies:
            foot_geoms.extend(foot.find_all('geom'))
        return tuple(foot_geoms)

    @property
    def prev_action(self):
        return self._prev_action


class PlanarWalkerObservables(legacy_base.WalkerObservables):
    """Observables for the PlanarWalker."""

    @composer.observable
    def appendages_pos(self):
        """Equivalent to `end_effectors_pos` with the head's position appended."""
        def appendages_pos_in_egocentric_frame(physics):
            appendages = self._entity.end_effectors
            appendages_xpos = physics.bind(appendages).xpos
            root_xpos = physics.bind(self._entity.root_body).xpos
            root_xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(
                np.dot(appendages_xpos - root_xpos, root_xmat), -1)
        return observable.Generic(appendages_pos_in_egocentric_frame)

    @composer.observable
    def bodies_quats(self):
        """Orientations of the bodies as quaternions, in the egocentric frame."""
        def bodies_orientations_in_egocentric_frame(physics):
            """Compute relative orientation of the bodies."""
            # Get the bodies
            bodies = self._entity.bodies
            # Get the quaternions of all the bodies &root in the global frame
            bodies_xquat = physics.bind(bodies).xquat
            root_xquat = physics.bind(self._entity.root_body).xquat
            # Compute the relative quaternion of the bodies in the root frame
            bodies_quat_diff = []
            for k in range(len(bodies)):
                bodies_quat_diff.append(
                    mjmath.mj_quatdiff(root_xquat, bodies_xquat[k]))  # q1^-1 * q2
            return np.reshape(np.stack(bodies_quat_diff, 0), -1)
        return observable.Generic(bodies_orientations_in_egocentric_frame)

    @composer.observable
    def bodies_pos(self):
        """Position of bodies relative to root, in the egocentric frame."""
        def bodies_pos_in_egocentric_frame(physics):
            """Compute relative orientation of the bodies."""
            # Get the bodies
            bodies = self._entity.bodies
            # Get the positions of all the bodies & root in the global frame
            bodies_xpos = physics.bind(bodies).xpos
            root_xpos, _ = self._entity.get_pose(physics)
            # Compute the relative position of the bodies in the root frame
            root_xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(
                np.dot(bodies_xpos - root_xpos, root_xmat), -1)
        return observable.Generic(bodies_pos_in_egocentric_frame)

    @property
    def proprioception(self):
        return ([self.joints_pos, self.joints_vel,
                 self.body_height, self.end_effectors_pos,
                 self.appendages_pos, self.world_zaxis,
                 self.bodies_quats, self.bodies_pos] +
                self._collect_from_attachments('proprioception'))
