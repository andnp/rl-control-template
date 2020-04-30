def isIterable(thing):
    try:
        _ = (x for x in thing)
        return True
    except:
        return False

def flatMap(f, gen):
    for x in gen:
        r = f(x)
        if isIterable(r):
            for y in r:
                yield y

        else:
            yield r
