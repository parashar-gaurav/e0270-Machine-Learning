import numpy as np
import matplotlib.pyplot as plt 
data = np.load('data.npy')
c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
for i in range(100):
	if i in c:
		plt.scatter(data[i][0],data[i][1],color='red')
	else:
		plt.scatter(data[i][0],data[i][1],color='green')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.savefig('4c.png')
