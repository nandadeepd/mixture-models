'''
Created by Nandadeep Davuluru
davuluru@pdx.edu

CS 546 - Advanced Machine learning, Spring 2018
Instructor - Anthony Rhodes

Following Expectation maximization to:
1. Implement K-means algorithm recursively. 
2. Gaussian Mixture Models 
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
ITERATIONS = 20
TETHA = 0.5


class GMM(object):

	def __init__(self, path, num_clusters, iterations):
		self.dataset = numpy.loadtxt(fname = path)
		self.colors = itertools.cycle(5 * ["r", "g", "c", "b", "k"])
		# assignments should be of the format -> cluster: [mean, cov, cluster points]
		self.assignments = dict()
		self.num_clusters = num_clusters
		self.loglikelihoods = list()
		self.priors = list()
		self.iterations = iterations


	def choosePoints(self):
		indices = [numpy.random.randint(0, self.dataset.shape[0]) for _ in range(self.num_clusters)]
		x, y = list(), list()
		for index in indices:
			x.append(self.dataset[index][0])
			y.append(self.dataset[index][1])
		self.centers = self.pointToList(x, y)
		return x, y



	# helper functions!
	getData = lambda self : self.dataset
	getll = lambda self : self.loglikelihoods
	pointToList = lambda self, x, y : [(xc, yc) for xc, yc in zip(x, y)]
	gaussian = lambda self, point, mean, covariance : multivariate_normal.pdf(point, mean, covariance)
	computeCovariance = lambda self, cluster : numpy.cov(cluster, rowvar = False)
	responsibility = lambda self, whichCluster : self.loglikelihoods[whichCluster]/sum(self.loglikelihoods)


	def getStuff(self, cluster):
		return cluster[0], cluster[1], cluster[2]

	# Not currently used. 
	def computeGaussianProbability(self, point, mean, covariance, prior_cluster):

		numerator = numpy.exp(-((point - mean).T * (point - mean) * (numpy.linalg.inv(covariance))) / 2)
		denominator = (numpy.sqrt(numpy.power((2 * numpy.pi), 2) * numpy.linalg.det(covariance)))
		# print("\n gaussian - {}".format(prior_cluster * (numerator / denominator)))
		pprint((numerator/denominator).shape)

		return prior_cluster * (numerator / denominator)

	
	# Init step after receiving clusters from K-means
	def initialization(self, assignments):

		for idx, (initCenters, initClusters) in enumerate(assignments.items()):
			initClusters = numpy.array(initClusters)
			initMean = ((sum(initClusters[:, 0]) / initClusters.shape[0]), ((sum(initClusters[:, 1]) / initClusters.shape[0])))
			initCovariance = self.computeCovariance(initClusters)
			assert(initCovariance.shape == (2,2))
			self.assignments[idx] = [initMean, initCovariance, initClusters]
		return self.assignments

	# computing log likelihoods
	def computeLogLikelihoods(self, assignments):

		for cluster in assignments.values():
			mean, covariance, points = self.getStuff(cluster)

			assert(covariance.shape == (2,2))

			prior = len(points) / len(self.dataset)
			self.priors.append(prior)
			# pprint("mean for cluster {}".format(mean))
			# pprint("prior for cluster {}".format(prior))
			likelihood = 0.0
			for point in points:
				likelihood += prior * self.gaussian(point, mean, covariance)
				# pprint("likelihood = {}".format(likelihood))
			self.loglikelihoods.append(numpy.log(likelihood))
		return self.loglikelihoods



	def findProbs(self, point, means, covariances):
		p = []
		for mean, covariance in zip(means, covariances):
			p.append(self.gaussian(point, mean, covariance))
		return p

	def plotClusters(self, assignments, iteration):
		for key, cluster in assignments.items():
			cluster = numpy.array(cluster)
			plotter.scatter(cluster[:, 0], cluster[:, 1], color = next(self.colors))
		# plotter.show()
		plotter.savefig("gmm" + str(iteration) + '.png')


	def run(self, loglikelihoods):
		converged = False
		iters = 1

		while not converged:
			new_assignments = dict()  # reflects the same format?

			old_means = [clusterProps[0] for clusterProps in self.assignments.values()]
			old_covariances = [clusterProps[1] for clusterProps in self.assignments.values()]

			for point in self.dataset:

				probability_point = self.findProbs(point, old_means, old_covariances)
				max_probab = numpy.argmax(probability_point)
				# pprint("testing insert index {}".format(numpy.argmax(probability_point)))

				if max_probab not in new_assignments:
					new_assignments[max_probab] = list()
				new_assignments[max_probab].append(point)

			self.plotClusters(new_assignments, iters)

			# compute error rates

			# send to init
			mod_ass = self.initialization(new_assignments)
			new_ll = self.computeLogLikelihoods(mod_ass)

			# pprint(new_ll)
			iters += 1

			if iters == self.iterations:
				converged = True


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

	gmm = GMM(FILENAME, CLUSTERS, ITERATIONS)

	initAss = gmm.initialization(assignments)

	gmm.computeLogLikelihoods(initAss)
	old_ll = gmm.getll() 
	gmm.run(old_ll)
	# pprint(gmm.getll())


if __name__ == '__main__':
	main()