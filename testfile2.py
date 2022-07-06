
def yield_function():
    a = range(10)
    for value in a:
        yield value


print(next(yield_function()))
