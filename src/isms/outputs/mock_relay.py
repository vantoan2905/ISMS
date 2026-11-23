"""In-process relay simulator."""


class MockRelay:
    def __init__(self, initial=False):
        self.is_on = initial

    def set_state(self, on):
        self.is_on = bool(on)
