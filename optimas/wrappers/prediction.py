from optimas.wrappers.example import Example

class Prediction(Example):
    """
    Adapted from the `dspy` library

    Represents a prediction object, extended from `Example` to support
    completions and language model usage statistics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self._demos
        del self._input_keys

        self._completions = None
        self._lm_usage = None

    def get_lm_usage(self):
        """Return language model usage metadata, if any."""
        return self._lm_usage

    def set_lm_usage(self, value):
        """Set language model usage metadata."""
        self._lm_usage = value

    @classmethod
    def from_completions(cls, list_or_dict, signature=None):
        """
        Create a Prediction from a list/dict of completions.
        Takes the first value of each completion as the default prediction.
        """
        obj = cls()
        obj._completions = Completions(list_or_dict, signature=signature)
        obj._store = {k: v[0] for k, v in obj._completions.items()}
        return obj

    @property
    def completions(self):
        """Access all completions associated with this prediction."""
        return self._completions

    def __repr__(self):
        store_repr = ",\n    ".join(f"{k}={v!r}" for k, v in self._store.items())
        if self._completions is None or len(self._completions) == 1:
            return f"Prediction(\n    {store_repr}\n)"
        return (
            f"Prediction(\n    {store_repr},\n"
            f"    completions=Completions(...)\n) "
            f"({len(self._completions) - 1} completions omitted)"
        )

    def __str__(self):
        return self.__repr__()

    def __float__(self):
        if "score" not in self._store:
            raise ValueError("Prediction object does not have a 'score' field to convert to float.")
        return float(self._store["score"])

    def _float_op(self, other, op):
        if isinstance(other, (int, float)):
            return op(self.__float__(), other)
        elif isinstance(other, Prediction):
            return op(self.__float__(), float(other))
        raise TypeError(f"Unsupported type for operation: {type(other)}")

    def __add__(self, other):
        return self._float_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._float_op(other, lambda a, b: b + a)

    def __truediv__(self, other):
        return self._float_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._float_op(other, lambda a, b: b / a)

    def __lt__(self, other):
        return self._float_op(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._float_op(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._float_op(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._float_op(other, lambda a, b: a >= b)


class Completions:
    """
    Represents multiple completions for a given prediction.
    Supports access by index or key, assuming aligned lists for each field.
    """

    def __init__(self, list_or_dict, signature=None):
        self.signature = signature

        if isinstance(list_or_dict, list):
            kwargs = {}
            for item in list_or_dict:
                for k, v in item.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict

        assert all(isinstance(v, list) for v in kwargs.values()), "All values must be lists"
        if kwargs:
            length = len(next(iter(kwargs.values())))
            assert all(len(v) == length for v in kwargs.values()), "All lists must have the same length"

        self._completions = kwargs

    def items(self):
        """Return (key, list-of-values) pairs."""
        return self._completions.items()

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")
            return Prediction(**{k: v[key] for k, v in self._completions.items()})
        return self._completions[key]

    def __getattr__(self, name):
        if name == "_completions":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._completions:
            return self._completions[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        """Return the number of completions."""
        return len(next(iter(self._completions.values())))

    def __contains__(self, key):
        return key in self._completions

    def __repr__(self):
        items_repr = ",\n    ".join(f"{k}={v!r}" for k, v in self._completions.items())
        return f"Completions(\n    {items_repr}\n)"

    def __str__(self):
        return self.__repr__()
