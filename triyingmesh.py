import numpy as np
# print(np.meshgrid(np.arange(100), np.arange(200)))
frames=[]
for i in range(10):
    frames.append([i,i,i,i])
i0=frames[:2]
i1=frames[2:]
print(frames[:-1])
print(frames[1:])
ix = [i0, i1]
print(ix)