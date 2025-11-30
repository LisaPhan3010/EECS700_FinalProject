import numpy as np

def binarize(features):
	# There are two ways I've found to binarize.
		# Compare to a median.
		# Compare by signs.
	# I'll try signs for now, but will adjust to median if needed.
	
	# Make sure that feature vector is Numpy array.
	feature_vec = np.asarray(features)
	
	# Sign method
	#binary = (feature_vec >= 0).astype(int)

	# Median method (in case we need it later)
	thresh = np.median(feature_vec)
	binary = (feature_vec >= thresh).astype(int)
	

	return binary

def hamming_distance(b1, b2):
	# Assuming binaries are same length, should be simple.
	# We ensure both binary vectors are numpy vectors.
	x = np.asarray(b1)
	y = np.asarray(b2)

	# We sum all differing values in x and y.
	dist = np.sum(x != y)

	# Then we'll normalize.
	return dist / len(x)

def build_score_matrix(probes, gallery):
	# Generate numpy score matrix.
	score_matrix = np.zeros((len(probes), len(gallery)))

	for i, p in enumerate(probes):
		for j, g in enumerate(gallery):
			score_matrix[i, j] = hamming_distance(p, g)

	return score_matrix
