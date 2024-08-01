class Policy:
    def __init__(self, input_space, policy) -> None:
        self.policy = policy
        self.input_space = input_space
        pass

    def get_policy(self, state):
        pass
    