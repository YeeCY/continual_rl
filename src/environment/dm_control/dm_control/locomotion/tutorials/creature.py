from dm_control import mjcf
import numpy as np

_BODY_RADIUS = 0.1
_HEIGHT = 0.15
_BODY_SIZE = [_BODY_RADIUS, _BODY_RADIUS, _BODY_RADIUS / 2]


class Leg:
    def __init__(self, length, rgba):
        self._model = mjcf.RootElement()

        # Defaults
        self._model.default.joint.damping = 2
        self._model.default.joint.type = 'hinge'
        self._model.default.geom.type = 'capsule'
        self._model.default.geom.rgba = rgba

        # Thigh
        self._thigh = self._model.worldbody.add('body')  # TODO: add name
        self._hip = self._thigh.add('joint', axis=[0, 0, 1])
        self._thigh.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[length / 4])

        # Shin
        self._shin = self._thigh.add('body', pos=[length, 0, 0])  # TODO: add name
        self._knee = self._shin.add('joint', axis=[0, 1, 0])
        self._shin.add('geom', fromto=[0, 0, 0, 0, 0, -length], size=[length / 4])

        # Position actuators
        self._model.actuator.add('position', joint=self._hip, kp=10)
        self._model.actuator.add('position', joint=self._knee, kp=10)

    @property
    def mjcf_model(self):
        return self._model


class Creature:
    def __init__(self, num_legs):
        self._model = mjcf.RootElement()
        self._model.compiler.angle = 'radian'
        rgba = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])

        # Torso
        self._torso = self._model.worldbody.add('body', name='torso')
        self._torso.add('geom', name='torso_geom', type='ellipsoid', size=_BODY_SIZE, rgba=rgba)

        # Attach legs
        for theta_idx in range(num_legs):
            theta = theta_idx * 2 * np.pi / num_legs
            leg_pos = _BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
            leg_site = self._torso.add('site', pos=leg_pos, euler=[0, 0, theta])
            leg = Leg(length=_BODY_RADIUS, rgba=rgba)
            leg_site.attach(leg.mjcf_model)

    @property
    def mjcf_model(self):
        return self._model


class CreatureArena:
    def __init__(self):
        self._model = mjcf.RootElement()
        chequered_tex = self._model.asset.add('texture', name='chequered_tex', type='2d', builtin='checker', width=300,
                                              height=300, rgb1=[0.2, 0.3, 0.4], rgb2=[0.3, 0.4, 0.5])
        grid_mat = self._model.asset.add('material', name='grid_mat', texture=chequered_tex,
                                         texrepeat=[5, 5], reflectance=0.2)

        # Gound and lights
        self._model.worldbody.add('geom', type='plane', size=[2, 2, 0.1], material=grid_mat)
        self._model.worldbody.add('light', pos=[-2, -1, 3], dir=[2, 1, -2])
        self._model.worldbody.add('light', pos=[2, -1, 3], dir=[-2, 1, -2])

        # Creatures
        self._creatures = [Creature(num_legs=num_legs) for num_legs in range(3, 9)]  # num_legs varies from 3 to 8

        # Place creatures
        height = _HEIGHT
        grid = 5 * _BODY_RADIUS
        xpos, ypos, zpos = np.meshgrid([-grid, 0, grid], [0, grid], [height])
        for idx, creature in enumerate(self._creatures):
            spawn_pos = [xpos.flat[idx], ypos.flat[idx], zpos.flat[idx]]
            spawn_site = self._model.worldbody.add('site', pos=spawn_pos, group=3)
            spawn_site.attach(creature.mjcf_model).add('freejoint')

    @property
    def mjcf_model(self):
        return self._model

    @property
    def creatures(self):
        return self._creatures
