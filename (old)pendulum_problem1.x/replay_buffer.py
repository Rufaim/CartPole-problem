import numpy as np

class ReplayBuffer(object):

	def __init__(self, buffer_size=-1):
		"""
		The right side of the buffer contains recent experiments
		If buffer_size equals to -1 than size is unlimited
		"""
		self.buffer_size = buffer_size
		self.size = 0
		self._s  = []
		self._a  = []
		self._r  = []
		self._t  = []
		self._s1 = []

	def __str__(self):
		if self.buffer_size > 0:
			total =  self.buffer_size
		else:
			total = "inf"
		return "ReplayBuffer size {} of {}".format(self.size,total)

	def add(self, s, a, r, t, s1):
		self._s.append(s)
		self._a.append(a)
		self._r.append(r)
		self._t.append(t)
		self._s1.append(s1)

		if self.size < self.buffer_size or self.buffer_size == -1: 
			self.size += 1
		else:
			self._s.pop(0)
			self._a.pop(0)
			self._r.pop(0)
			self._t.pop(0)
			self._s1.pop(0)
		return self

	def extend(self, S, A, R, T, S1):
		L = len(S)
		assert L == len(A) and L == len(R) and L == len(T) and L==len(S1)

		self._s.extend(S)
		self._a.extend(A)
		self._r.extend(R)
		self._t.extend(T)
		self._s1.extend(S1)

		if self.size+L < self.buffer_size or self.buffer_size == -1: 
			self.size += L
		else:
			L =  L - self.buffer_size
			self._s = self._s[L:]
			self._a = self._a[L:]
			self._r = self._r[L:]
			self._t = self._t[L:]
			self._s1 = self._s1[L:]
			self.size = self.buffer_size
		return self

	def sample_batch(self, batch_size):
		batch = min(self.size,batch_size)
		idx = np.random.randint(0,self.size,(batch))

		s_batch,a_batch,r_batch,t_batch,s1_batch = [],[],[],[],[]
		for i in idx:
			s_batch.append(self._s[i])
			a_batch.append(self._a[i])
			r_batch.append(self._r[i])
			t_batch.append(self._t[i])
			s1_batch.append(self._s1[i])

		return np.array(s_batch), np.array(a_batch), np.array(r_batch), np.array(t_batch), np.array(s1_batch)

	def clear(self):
		self.size = 0
		self._s.clear()
		self._a.clear()
		self._r.clear()
		self._t.clear()
		self._s1.clear()
		return self
