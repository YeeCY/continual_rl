from dm_control import mjcf
from dm_control.locomotion.tutorials.creature import CreatureArena
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def record_video_and_trajectory(physics, arena, duration=10, framerate=30, video_name='creature_video.avi',
                                trajectory_name='trajectory.jpg'):
    frames = []
    pos_x = []
    pos_y = []
    creature_torsos = []
    creature_actuators = []
    for creature in arena.creatures:
        creature_torsos.append(creature.mjcf_model.find('geom', 'torso_geom'))
        creature_actuators.extend(creature.mjcf_model.find_all('actuator'))

    # Sinusoidal control signals
    freq = 5
    phase = 2 * np.pi * np.random.rand(len(creature_actuators))
    amp = 0.9

    # Step simulations
    physics.reset()
    while physics.data.time < duration:
        physics.bind(creature_actuators).ctrl = amp * np.sin(freq * physics.data.time + phase)
        physics.step()

        # Save creature positions
        pos_x.append(physics.bind(creature_torsos).xpos[:, 0].copy())
        pos_y.append(physics.bind(creature_torsos).xpos[:, 1].copy())

        # Save video frames
        if len(frames) < physics.data.time * framerate:
            pixels = physics.render()
            frames.append(pixels)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(video_name, fourcc, framerate, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

    creature_colors = physics.bind(creature_torsos).rgba[:, :3]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_prop_cycle(color=creature_colors)
    _ = ax.plot(pos_x, pos_y, linewidth=4)
    plt.savefig(trajectory_name)

    return frames


def main():
    arena = CreatureArena()

    # Instantiate the physics and render
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    image = Image.fromarray(physics.render())
    image.save('creature_arena.jpg')

    record_video_and_trajectory(physics, arena)


if __name__ == "__main__":
    main()
