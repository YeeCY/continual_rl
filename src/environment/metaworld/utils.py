import random


def round_robin_strategy(num_tasks, last_task=None):
    """A function for sampling tasks in round robin fashion.
    Args:
        num_tasks (int): Total number of tasks.
        last_task (int): Previously sampled task.
    Returns:
        int: task id.
    """
    if last_task is None:
        return 0

    return (last_task + 1) % num_tasks


def uniform_random_strategy(num_tasks, _):
    """A function for sampling tasks uniformly at random.
    Args:
        num_tasks (int): Total number of tasks.
        _ (object): Ignored by this sampling strategy.
    Returns:
        int: task id.
    """
    return random.randint(0, num_tasks - 1)
