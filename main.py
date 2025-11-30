import numpy as np
from features import image_features
from binarization import binarize, build_score_matrix
from threshold import select_threshold, authentication, auth_result
def main():
	'''
	Filler inputs for now
	These can be removed when we have the actual feature vectors.
	'''
	'''gabor_P5 = np.random.randn(128)
	gabor_P6 = np.random.randn(128)
	gabor_P7 = np.random.randn(128)

	gabor_G1 = np.random.randn(128)
	gabor_G2 = np.random.randn(128)
	gabor_G3 = np.random.randn(128)
	gabor_G4 = np.random.randn(128)'''
    # Feature vectors from feature extraction process  
	probes_law  = [image_features['FP5'], image_features['FP6'], image_features['FP7']]
	gallery_law = [image_features['FP1'], image_features['FP2'], image_features['FP3'], image_features['FP4']]

	# Step 1: Likely some file I/O stuff.
	
	# Step 2: Feature extraction	

	# Step 3: Binarization
	# probes = [binarize(x) for x in [gabor_P5, gabor_P6, gabor_P7]]
	# gallery = [binarize(x) for x in [gabor_G1, gabor_G2, gabor_G3, gabor_G4]]	
	probes = [binarize(x) for x in probes_law]
	gallery = [binarize(x) for x in gallery_law]

	# Step 4: Build Score Matrix
	scores = build_score_matrix(probes, gallery)
	print("Score Matrix:")
	print(scores)

	# Step 5: Authentication Process
	threshold = select_threshold(scores)
	auth_matrix, match_pairs = authentication(scores, threshold)
	auth_result(auth_matrix, match_pairs, threshold)

main()