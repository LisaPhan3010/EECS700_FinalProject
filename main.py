import numpy as np
from binarization import binarize, build_score_matrix

def main():
	'''
	Filler inputs for now
	These can be removed when we have the actual feature vectors.
	'''
	gabor_P5 = np.random.randn(128)
	gabor_P6 = np.random.randn(128)
	gabor_P7 = np.random.randn(128)

	gabor_G1 = np.random.randn(128)
	gabor_G2 = np.random.randn(128)
	gabor_G3 = np.random.randn(128)
	gabor_G4 = np.random.randn(128)
	# Step 1: Likely some file I/O stuff.
	
	# Step 2: Feature extraction	

	# Step 3: Binarization
	probes = [binarize(x) for x in [gabor_P5, gabor_P6, gabor_P7]]
	gallery = [binarize(x) for x in [gabor_G1, gabor_G2, gabor_G3, gabor_G4]]	

	# Step 4: Build Score Matrix
	scores = build_score_matrix(probes, gallery)
	print(scores)

	# Step 5: Authentication Process

main()
