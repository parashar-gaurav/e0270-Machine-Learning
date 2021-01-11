import numpy as np
import dill

###part a
##importing kernel functions


def symm_checker(A,tolerance=1e-5):
	flag = True
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			if np.abs(A[i][j]-A[j][i])>tolerance:
				flag = False
	return flag

def eigvalues_calc(A,tolerance=1e-6):
	#input to function eigvalsh should be a real symmetric matrix
	e = np.linalg.eigvalsh(A)
	print('min eigval = ',min(e))
	if min(e) > -tolerance:
		return True
	else:
		return False

def isPSD(A, tol=1e-16):
  E = np.linalg.eigvalsh(A)
  return np.all(E > -tol)

with open('function1.pkl','rb') as filename:
	f1 = dill.load(filename)
	f1 = dill.loads(f1)

with open('function2.pkl','rb') as filename:
	f2 = dill.load(filename)
	f2 = dill.loads(f2)

with open('function3.pkl','rb') as filename:
	f3 = dill.load(filename)
	f3 = dill.loads(f3)


with open('function4.pkl','rb') as filename:
	f4 = dill.load(filename)
	f4 = dill.loads(f4)
	
####setting n = 100

n = 100

####sampling 100 x where x belong to R3 , sampling each component uniformly from [-5,+5]
x1 = []
x2 = []
x3 = []

for i in range(n):
	x1.append(np.random.uniform(-5,5))
	x2.append(np.random.uniform(-5,5))
	x3.append(np.random.uniform(-5,5))
	
###let K_i be the matrix for kernel function i 

K1 = np.zeros((n,n))
K2 = np.zeros((n,n))
K3 = np.zeros((n,n))
K4 = np.zeros((n,n))

##filling up all the matrices:

for i in range(n):
	for j in range(n):
		x = np.array([[x1[i]],[x2[i]],[x3[i]]])
		y = np.array([[x1[j]],[x2[j]],[x3[j]]])
		K1[i][j] = f1(x,y)
		K2[i][j] = f2(x,y)
		K3[i][j] = f3(x,y)
		K4[i][j] = f4(x,y)

#print('matrix1 = ',K1)

#print('matrix1 = ',K2)

#print('matrix1 = ',K3)

#print('matrix1 = ',K4)

#print(eigvalues_calc(K1))
#print(eigvalues_calc(K2))
#print(eigvalues_calc(K3))
#print(eigvalues_calc(K4))

#print('checking symmetric : ')
#print(symm_checker(K1))
#print(symm_checker(K2))
#print(symm_checker(K3))
#print(symm_checker(K4))

if(symm_checker(K1)):
	if(eigvalues_calc(K1)):
		print('K1 is a symmetric and positive semi-definite matrix. Might be a valid kernel!')
	else:
		print('K1 is symmetric but not positive semi-definite matrix. Definitely not a valid kernel!')
else:
	print('K1 is not a symmetric matrix. Not a valid Kernel!')


if(symm_checker(K2)):
	if(eigvalues_calc(K2)):
		print('K2 is a symmetric and positive semi-definite matrix. Might be a valid kernel!')
	else:
		print('K2 is symmetric but not positive semi-definite matrix. Definitely not a valid kernel!')
else:
	print('K2 is not a symmetric matrix. Not a valid Kernel!')
	
	
if(symm_checker(K3)):
	if(eigvalues_calc(K3)):
		print('K3 is a symmetric and positive semi-definite matrix. Might be a valid kernel!')
	else:
		print('K3 is symmetric but not positive semi-definite matrix. Definitely not a valid kernel!')
else:
	print('K3 is not a symmetric matrix. Not a valid Kernel!')



if(symm_checker(K4)):
	if(eigvalues_calc(K4)):
		print('K4 is a symmetric and positive semi-definite matrix. Might be a valid kernel!')
	else:
		print('K4 is symmetric but not positive semi-definite matrix. Definitely not a valid kernel!')
else:
	print('K4 is not a symmetric matrix. Not a valid Kernel!')
	
'''
def check_identity(A):
	flag = True
	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			if (i==j):
				if (A[i][i]==0.0):
					flag = False
			else:
				if (A[i][j]==1.0):
					print(i,j)
					flag = False
	return flag
	
print(check_identity(K4))
			
eye = np.eye(100)
a = eye == K4
print(a)
print(a.all())
'''
#####################3code to check function 5###############
with open('function5.pkl','rb') as filehandle:
	f5 = dill.load(filehandle)
	f5 = dill.loads(f5)

#loading the vector sampler function
with open('k5sampler.pkl','rb') as filehandle:
	sampler = dill.load(filehandle)
	sampler = dill.loads(sampler)
#sampling 100 points:
x5 = []
K5 = np.zeros((n,n))
for i in range(n):
	x5.append(sampler())

for i in range(n):
	for j in range(n):
		K5[i][j] = f5(x5[i],x5[j])

if(symm_checker(K5)):
	if(eigvalues_calc(K5)):
		print('K5 is a symmetric and positive semi-definite matrix. Might be a valid kernel!')
	else:
		print('K5 is symmetric but not positive semi-definite matrix. Definitely not a valid kernel!')
else:
	print('K5 is not a symmetric matrix. Not a valid Kernel!')
	

