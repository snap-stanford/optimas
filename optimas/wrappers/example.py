class Example:
    """
    Adapted from the `dspy` library

    A lightweight wrapper around a dictionary that supports attribute access,
    input/label separation, and deep copying.
    """
    def __init__(self, base=None, **kwargs):
        self._store = {}
        self._demos = []
        self._input_keys = None

        if base and isinstance(base, type(self)):
            self._store = base._store.copy()
        elif base and isinstance(base, dict):
            self._store = base.copy()

        self._store.update(kwargs)

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError
        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("_") or key in dir(self.__class__):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        return len([k for k in self._store if not k.startswith("dspy_")])

    def __repr__(self):
        d = {k: v for k, v in self._store.items() if not k.startswith("dspy_")}
        return f"Example({d}) (input_keys={self._input_keys})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Example) and self._store == other._store

    def __hash__(self):
        return hash(tuple(self._store.items()))

    def keys(self, include_dspy=False):
        """Return keys in the example."""
        return [k for k in self._store if include_dspy or not k.startswith("dspy_")]

    def values(self, include_dspy=False):
        """Return values in the example."""
        return [v for k, v in self._store.items() if include_dspy or not k.startswith("dspy_")]

    def items(self, include_dspy=False):
        """Return (key, value) pairs in the example."""
        return [(k, v) for k, v in self._store.items() if include_dspy or not k.startswith("dspy_")]

    def get(self, key, default=None):
        return self._store.get(key, default)

    def with_inputs(self, *keys):
        """Return a copy of the example with specified input keys."""
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self):
        """Return a new Example instance with only the input keys."""
        if self._input_keys is None:
            raise ValueError("Inputs have not been set. Use `example.with_inputs(...)` first.")
        d = {key: self._store[key] for key in self._input_keys if key in self._store}
        new_instance = type(self)(base=d)
        new_instance._input_keys = self._input_keys
        return new_instance

    def labels(self):
        """Return a new Example instance with all non-input keys."""
        input_keys = self.inputs().keys()
        d = {key: self._store[key] for key in self._store if key not in input_keys}
        return type(self)(base=d)

    def __iter__(self):
        return iter(dict(self._store))

    def copy(self, **kwargs):
        """Return a shallow copy of the example, with optional updates."""
        return type(self)(base=self, **kwargs)

    def without(self, *keys):
        """Return a copy of the example with specific keys removed."""
        copied = self.copy()
        for key in keys:
            copied._store.pop(key, None)
        return copied

    def to_dict(self):
        """Return a full dictionary copy of the internal store."""
        return self._store.copy()
