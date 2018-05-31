'''
Created by Nandadeep Davuluru
davuluru@pdx.edu

CS 546 - Advanced Machine learning, Spring 2018
Instructor - Anthony Rhodes

Implementing K-means algorithm recursively. 
'''
import numpy, sys, itertools
import matplotlib.pyplot as plotter
from scipy.spatial import distance
from pprint import pprint
from scipy.stats import multivariate_normal


# Globals, hyperparameters

FILENAME = 'GMM_dataset.txt'
CLUSTERS = 5
PLOTS = './'


class GMM(object):

	def __init__(self, path, num_clusters):
		self.dataset = numpy.loadtxt(fname = path)
		self.colors = itertools.cycle(5 * ["r", "g", "c", "b", "k"])
		# assignments should be of the format -> cluster: [mean, cov, cluster points]
		self.assignments = dict()
		self.num_clusters = num_clusters

	def getData(self):
		return self.dataset

	def choosePoints(self):
		indices = [numpy.random.randint(0, self.dataset.shape[0]) for _ in range(self.num_clusters)]
		x, y = list(), list()
		for index in indices:
			x.append(self.dataset[index][0])
			y.append(self.dataset[index][1])
		self.centers = self.pointToList(x, y)
		return x, y

	# computeCovariance = lambda self, one, two : numpy.cov(one, two)
	pointToList = lambda self, x, y : [(xc, yc) for xc, yc in zip(x, y)]

	def computeGaussianProbability(self, point, mean, covariance):

		# pprint("mean: {}".format(mean))
		# pprint("covariance: {}".format(covariance.shape))
		# pprint("point: {}".format(point))

		numerator = numpy.exp(-((point - mean).T * (point - mean) * (numpy.linalg.inv(covariance))) / 2)
		denominator = (numpy.sqrt(numpy.power((2 * numpy.pi), 2) * numpy.linalg.det(covariance)))
		return numerator / denominator

	computeGaussian = lambda self, point, mean, covariance : multivariate_normal.pdf(mean, covariance, point)

	def computeClusterMean(self, cluster):
		# print(type(cluster), cluster)
		cluster = numpy.array(cluster)
		Xmean = sum(cluster[:, 0]) / cluster.shape[0]
		Ymean = sum(cluster[:, 1]) / cluster.shape[0]
		return (Xmean, Ymean)


	computeCovariance = lambda self, cluster : numpy.cov(cluster, rowvar = False)

	def iterClusters(self, assignments):

		centers = list(assignments.keys())
		for center, cluster in assignments.items():
			mean = self.computeClusterMean(cluster)
			covariance = self.computeCovariance(cluster)
			for point in cluster:

				center_probabilities = [self.computeGaussianProbability(point, mean, covariance) for _ in centers]
				max_center_probability = numpy.argmax(center_probabilities)
				pprint(centers[max_center_probability])

				# now do the assignments. 



class KMeans(object):

	def __init__(self, path, k):

		# space blowout..
		self.dataset = numpy.loadtxt(fname = path)
		self.K = k
		self.x_range, self.y_range = max(self.dataset[:, 0]), max(self.dataset[:, 1])
		self.assignments = dict()
		self.centers = list()
		self.colors = itertools.cycle(5 * ["r", "g", "c", "b", "k"])
		self.prevCenters = list()

	def setPrevCenters(self, centers):
		self.prevCenters = centers

	def getAssignments(self):
		return self.assignments

	def generatePoints(self):
		xList, yList = list(), list()
		for _ in range(self.K):
			xList.append(numpy.random.randint(0, self.x_range))
			yList.append(numpy.random.randint(0, self.y_range))
			# centroids.append((x, y))
		return xList, yList

	def choosePoints(self):
		indices = [numpy.random.randint(0, self.dataset.shape[0]) for _ in range(CLUSTERS)]
		x, y = list(), list()
		for index in indices:
			x.append(self.dataset[index][0])
			y.append(self.dataset[index][1])
		self.centers = self.pointToList(x, y)
		return x, y

	def iterPoints(self, ccenters):
		# assignments = dict()
		for point in self.dataset:
			distVec = [distance.euclidean(point, center) for center in ccenters]
			# finding the min of distVec to assign point to a center
			min_dist = numpy.argmin(distVec)

			if ccenters[min_dist] not in self.assignments:
				self.assignments[ccenters[min_dist]] = list()
			self.assignments[ccenters[min_dist]].append(point)
		return self.assignments

	def computeCentroid(self, cluster):
		cluster = numpy.array(cluster)
		Xmean = sum(cluster[:, 0]) / cluster.shape[0]
		Ymean = sum(cluster[:, 1]) / cluster.shape[0]
		return (Xmean, Ymean)


	def plotCluster(self, centers, clusters, iteration):
		centers = numpy.array(centers)
		for cluster in clusters:
			cluster = numpy.array(cluster)
			plotter.scatter(cluster[:, 0], cluster[:, 1], c = next(self.colors))
		plotter.scatter(centers[:, 0], centers[:, 1], color = 'yellow', marker = '*', label = iteration)
		# plotter.show()
		plotter.savefig(PLOTS + str(iteration) + '.png')


	pointToList = lambda self, x, y : [(xc, yc) for xc, yc in zip(x, y)]
	errorRate = lambda self, oldCenters, newCenters : numpy.linalg.norm(numpy.array(oldCenters) - numpy.array(newCenters), axis = 1)

	def run(self, iters, centers, clusters):
		'''
		Main algorithm to find K clusters for N data points. 
		Recursively find new centroids until base cases are hit. 
		'''
		if iters == self.K:
			return "\nIterations done.\n"

		if not iters == 1:
			if self.errorRate(centers, self.prevCenters).any() == 0:
				return "\nConverged.\n"
		new_centroids = list()
		# self.plotCluster(centers, clusters, iters)

		for center, cluster in zip(centers, clusters):
			new_centroid = self.computeCentroid(cluster)
			new_centroids.append(new_centroid)
		iters += 1
		return self.run(iters, numpy.array(new_centroids), clusters)
	


def main():


	kmeans = KMeans(FILENAME, CLUSTERS)	
	x, y = kmeans.choosePoints()
	centers = kmeans.pointToList(x, y)
	assignments = kmeans.iterPoints(centers)
	clusters = assignments.values()

	# initial center points. 
	kmeans.setPrevCenters(centers)
	message = kmeans.run(1, centers, clusters)

	assignments = kmeans.getAssignments()

	print("The stopping reason: {}".format(message))

	gmm = GMM(FILENAME, 5)
	# dataset = gmm.getData()
	x, y = gmm.choosePoints()
	gmm.iterClusters(assignments)

	# send these clusters and centers to GMM.







if __name__ == '__main__':
	main()


