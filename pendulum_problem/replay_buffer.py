import numpy as np

class ReplayBuffer(object):
    """Classical Replay Buffer storing SARS tuples.
    The right side of the buffer contains recent experiments

    Parameters:
    ----------
    buffer_size : int (default -1)
        Maximum buffer capacity. If equals to -1 than size is unlimited.
    seed : int (default None)
        Seed to sample minibatches.
    """
    def __init__(self, buffer_size=-1, seed=None):
        self.buffer_size = buffer_size
        self._random_generator = np.random.RandomState(seed)
        self._s  = []
        self._a  = []
        self._r  = []
        self._t  = []
        self._s1 = []

    def __str__(self):
        if self.buffer_size > 0:
            return "ReplayBuffer size {} of {}".format(self.size(),self.buffer_size)
        else:
            return f"ReplayBuffer size {self.size()}"

    def size(self):
        """Returns current buffer size
        """
        return len(self._s)

    def add(self, s, a, r, t, s1):
        """Pushes a SARS-tuple to buffer
        """
        self._s.append(s)
        self._a.append(a)
        self._r.append(r)
        self._t.append(t)
        self._s1.append(s1)

        if self.buffer_size > 0 and self.size() >= self.buffer_size :
            self._s.pop(0)
            self._a.pop(0)
            self._r.pop(0)
            self._t.pop(0)
            self._s1.pop(0)

        return self

    def sample_batch(self, batch_size):
        """Returns minibatch sampled from the buffer with replacments.
        """
        batch = min(self.size(),batch_size)
        idx = self._random_generator.randint(0,self.size(),(batch,))

        s_batch,a_batch,r_batch,t_batch,s1_batch = [],[],[],[],[]
        for i in idx:
            s_batch.append(self._s[i])
            a_batch.append(self._a[i])
            r_batch.append(self._r[i])
            t_batch.append(self._t[i])
            s1_batch.append(self._s1[i])

        return np.array(s_batch,dtype=np.float32), np.array(a_batch,dtype=np.float32), np.array(r_batch,dtype=np.float32),\
               np.array(t_batch,dtype=np.bool), np.array(s1_batch,dtype=np.float32)

    def clear(self):
        self._s.clear()
        self._a.clear()
        self._r.clear()
        self._t.clear()
        self._s1.clear()
        return self
