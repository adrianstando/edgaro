import functools

print_unbuffered = functools.partial(print, flush=True)
