"""Lookup of tracker implementations by name."""


class TrackerRegistry:
    def __init__(self):
        self._classes = {}

    def register(self, name):
        def register_class(cls):
            if name in self._classes:
                raise ValueError(f"Tracker already registered: {name}")
            self._classes[name] = cls
            return cls

        return register_class

    def has(self, name):
        return name in self._classes

    def get(self, name):
        return self._classes[name]


TRACKER_REGISTRY = TrackerRegistry()
