from dm_control import viewer
from dm_control import suite


def main():
    env = suite.load(domain_name="walker", task_name="walk")
    viewer.launch(environment_loader=env)


if __name__ == "__main__":
    main()
