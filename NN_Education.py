import math
import numpy


# Define sigmoid function
def sigmoid(x):
	return 1/(1 + math.exp(x*(-1)) )
#print sigmoid(.5)


# Import training dataset
education_train = numpy.genfromtxt('education_train.csv',dtype=None, delimiter=',',skip_header=1)
#print education_train


# Initialize an array for holding weights, with random numbers
weights = numpy.random.rand(len(education_train[0]),1)/10000
#print weights


# Add a column with 1's and divide by 100 to get a value from 0 to 1
education_train2 = numpy.zeros( (len(education_train),len(education_train[0])+1) )
for i in range(0,len(education_train2)):
	for j in range(0,len(education_train2[0])):
		if j == 0:
			education_train2[i][j] = 1
		else:
			education_train2[i][j] = education_train[i][j-1]/100

#print education_train2			

# Define function to calculate the output of sigmoid unit
def sig_out(x,y):
# Drop the target column from x(training data) 
	x2 = (x.T[0:(len(x.T)-1)]).T
# Calculate the vectorized input for sigmoid function	
	sigmoid_input = numpy.dot(x2,y)
# Use Sigmoid function on sigmoid input
	return numpy.vectorize(sigmoid)(sigmoid_input)

#print sig_out(education_train2,weights)	
	
"""
gradient = numpy.zeros( (len(education_train[0])+1,1) )
	
a = ((education_train2[:,len(education_train2[0])-1] - sig_out(education_train2,weights).T)*sig_out(education_train2,weights).T*(1-sig_out(education_train2,weights)).T)
print sum(education_train2*a.T)
print gradient.T +  sum(education_train2*a.T)
print gradient
print (gradient.T +  sum(education_train2*a.T)).T
"""


# Define the cost function
def costf(x,y):
# Separate target column
	target = x[:,len(x[0])-1]
#Calculate the error	
	err = target - sig_out(x,y).T
	return (0.5)*sum(sum(err*err))
	
	
#print costf(education_train2,weights)



# Code to learn weights
def learn_weights(x,y):

	# Drop the target column from x(training data)
	x2 = (x.T[0:(len(x.T)-1)]).T
	
	#Initialize learning rate 'r'
	r = 0.01
	
	#print weights
	#print costf(x,y)
	
	for i in range(1,1000):
		gradient = sum( x2*( ((x[:,len(x[0])-1] - sig_out(x,y).T) * sig_out(x,y).T * (1-sig_out(x,y)).T ).T ) )
		y = (y.T + r*gradient).T
	
	#print y
	#print costf(x,y)
	return y

#learn_weights(education_train2,weights)


# Import test dataset
education_test = numpy.genfromtxt('education_dev.csv',dtype=None, delimiter=',',skip_header=1)

# Add a column with 1's and divide by 100 to get a value from 0 to 1
education_test2 = numpy.zeros( (len(education_test),len(education_test[0])+1) )
for i in range(0,len(education_test2)):
	for j in range(0,len(education_test2[0])):
		if j == 0:
			education_test2[i][j] = 1
		else:
			education_test2[i][j] = education_test[i][j-1]/100

# Code to score test dataset
print sig_out(education_test2,learn_weights(education_train2,weights))	