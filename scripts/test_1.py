def func2(a, b, c):
    print(a, b, c)
    return a + b + c

def func(kwargs=None):
    defaults = {"a": 1, "b": 2, "c": 3}
    kwargs = {} if kwargs is None else kwargs
    kwargs = {**defaults, **kwargs}
    return func2(**kwargs)


func({"d": 5})
