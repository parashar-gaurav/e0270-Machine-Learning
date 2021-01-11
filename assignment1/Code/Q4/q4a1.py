import numpy as np
import dill
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#np.random.seed(15)
import copy

#####################part a#####################
################################################

with open('kernel_4a.pkl','rb') as filehandle:
	kernelf = dill.load(filehandle)
	kernelf = dill.loads(kernelf)

# distance function

def dist(x,y,k):
	'''takes input two vectors x and y of dimensions (d,1) and the kernel function name; returns the kernel function value'''
	#||a-b||^2 = (a-b).T*(a-b) = (a.T - b.T)*(a-b) = a.Ta + b.Tb - 2a.Tb
	#distance = sqrt(aTa + bTb - 2aTb)
	#kernel function calculates k(x,y) = phi(x).Tphi(y)
	sq_dist = k(x,x) + k(y,y) - 2*k(x,y)
	#for euclidean distance
	#sq_dist = np.sum((x-y)**2)
	
	return(np.sqrt(sq_dist))

def part_a():
	'''calculates distance matrix D of 10x10 Identity matrix'''
	d = 10
	E = np.empty((d,0))
	#print(E.shape)
	#print(E)
	for i in range(d):
		e_i = np.zeros((d,1))
		e_i[i] = 1
		#print(e_i.shape)
		#print(e_i)
		E = np.hstack((E,e_i))
	#E[2][4] = 6
	#print(E)
	#print(E[:][2])
	D = np.zeros((d,d))
	for i in range(d):
		for j in range(d):
			D[i][j] = dist(E[:,[i]],E[:,[j]],kernelf)
	
	print("The matrix D is : \n",D)
	elemsum = 0.0
	for i in range(d):
		for j in range(d):
			elemsum = elemsum + D[i][j]
	
	print("Sum of all the entries in D is :",elemsum)
	

#########################part b###################
##################################################


def dist_b(arrX,i):
	#takes an array containing all the e_i's and integer value i , and returns distance between phi(e_i) and mu
	#dist(phi(e_i),mu) = phi(e_i).Tphi(e_i) + mu.T(mu) -2*phi(e_i).T(mu)
	#mu = 1/d (phi(e_1)+phi(e_2)+...+phi(e_d))
	#mu.T(mu) = (phi1 + phi2 + ... +phid).T(phi1 + phi2 + ... + phid)/d^2 = 0 as calculated in part (a)
	# phi(e_i).T (mu) = [phi(e_i).Tphi(e_1) + phi(e_i).T(phi(e_2)) + ... + phi(e_i).T(phi(e_d))]/d = 0 as all entries of D is 0
	

	#E[2][4] = 6
	#print(E)
	#print(E[:][2])
	d = 10
	D = np.zeros((d,d))
	for j in range(d):
		for k in range(d):
			D[j][k] = dist(arrX[:,[j]],arrX[:,[k]],kernelf)
	
	#print("The matrix D is : \n",D)
	elemsum = 0.0
	for j in range(d):
		for k in range(d):
			elemsum = elemsum + D[j][k]
	
	phi_iTphi_i = D[i][i]
	phi_iTmu = 0.0
	for j in range(d):
		phi_iTmu = phi_iTmu + D[i][j]
	
	phi_iTmu = phi_iTmu/d
	muTmu = elemsum/(d**2)			
	res = phi_iTphi_i + muTmu -2*phi_iTmu
	return(np.sqrt(res))

def part_b():
	d = 10
	E = np.empty((d,0))
	#print(E.shape)
	#print(E)
	for i in range(d):
		e_i = np.zeros((d,1))
		e_i[i] = 1
		#print(e_i.shape)
		#print(e_i)
		E = np.hstack((E,e_i))
	
	sumval = 0.0
	for i in range(d):
		sumval = sumval + dist_b(E,i)
	
	print('sum of all d_i is : ',sumval)





#######################part c######################
###################################################

data = np.load('data.npy')
#print(data)
D = np.zeros((100,100))
for j in range(100):
	for k in range(100):
		j_vec = data[j].reshape(2,1)
		k_vec = data[k].reshape(2,1)
		a = kernelf(j_vec,j_vec)
		b = kernelf(k_vec,k_vec)
		c = kernelf(j_vec,k_vec)
		D[j][k] = np.sqrt(a+b-2*c)
#print(D[1][99])
phi = np.zeros((100,100),dtype=float)
for j in range(100):
	for k in range(100):
		j_vec = data[j].reshape(2,1)
		k_vec = data[k].reshape(2,1)
		a = kernelf(j_vec,k_vec)
		phi[j][k] = a
		
#for j in range(100):
#print(phi[j])
#print(D[20][75])
#print(D[75][20])
def dist_calc(j,mu_list):
	term1 = phi[j][j]
	temp_list1 = mu_list
	temp_list2 = mu_list
	term2 = 0.0
	for l in temp_list1:
		for m in temp_list2:
			term2 = term2 + phi[l][m]
	if(len(mu_list)!=0):
		term2 = term2/(len(mu_list)**2)
	term3 = 0.0
	temp_list3 = mu_list
	for i in temp_list3:
		term3 = term3 + phi[j][i]
	if(len(mu_list)!=0):	
		term3 = term3/len(mu_list)
	return (term1 + term2 - 2*term3)

def part_c():

	
	mu1,mu2 = np.random.choice(data.shape[0], 2, replace=False)
	#mu1 = mu1.reshape(2,1)
	#mu2 = mu2.reshape(2,1)
	#print(mu1)
	#print(mu2)
	m1 = [mu1]
	m2 = [mu2]
	c1_old = []
	c2_old = []
	label = [None]*100
	for i in range(100):

		c1 = []
		c2 = []
		for j in range(100):
			#dist1 = D[j][mu1]
			#dist2 = D[j][mu2]
			dist1 = dist_calc(j,m1)
			dist2 = dist_calc(j,m2)
			#print(' j = ',j,'dist1 : ',dist1,'dist2 :',dist2)
			if dist1 <= dist2:
				
				label[j] = 1
				c1.append(j)
			else:
				
				label[j] = 2
				c2.append(j)

		#print('c1',c1)
		#print('c2',c2)
		mu1_new = np.zeros((2,1))
		mu2_new = np.zeros((2,1))
		
		m1 = c1
		m2 = c2
		
		if(c1==c1_old):
			break
		c1_old = copy.deepcopy(c1)
		c2_old = copy.deepcopy(c2)
	print('c1',c1)
	print('c2',c2)
		
	#print(label)
	#for i in range(100):
		#plt.scatter(data[i][0],data[i][1],color='blue')
	#plt.show()
	#print(c1,c2)
			
	
part_a()
part_b()
part_c()




