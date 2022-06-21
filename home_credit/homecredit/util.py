import time
from contextlib import contextmanager # with를 사용할 수 있게 해줌

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))