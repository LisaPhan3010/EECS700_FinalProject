import numpy as np
import os
import cv2
from PIL import Image
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from skimage.filters import gabor_kernel

#==============================#
#=========BINARIZATION=========#
#==============================#

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

#==============================#
#==========THRESHOLD===========#
#==============================#

def select_threshold(score_matrix):
    '''
    Automatically select a threshold based on score matrix.
    I use median for all scores. 
    '''
    threshold = np.median(score_matrix) 
    return threshold

def authentication(score_matrix, threshold):
    '''
    In order to authencation, I compared each score to the threshold
    Steps:
    1. Create a binary matrix where 1 indicates a match (score <= threshold)
    2. Loop through each probe-gallery pair (if distance < threshold, it's a match)'''
    # Create a binary authentication matrix based on the threshold
    auth_matrix = (score_matrix <= threshold).astype(int)

    match_pairs = []
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            if score_matrix[i, j] < threshold:
                match_pairs.append((i, j, score_matrix[i, j]))
    return auth_matrix, match_pairs

def auth_result(auth_matrix, match_pairs, threshold):
    print(f"Threshold used: {threshold:.5f}")
    print("Authentication Matrix:")
    print(auth_matrix)
    print("Matched Pairs (Probe Index, Gallery Index, Score < threshold):")
    if len(match_pairs)== 0:
        print("No matches found.")
    else:
        for probe, gallery, score in match_pairs:
            print(f"Probe {probe} - Gallery {gallery} : Score = {score:.5f}")

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
	print("Probes:")
	for x in range(len(probes)):
		print(f'FP{x+5}: {probes[x]}')
	#[print(x) for x in probes]
	gallery = [binarize(x) for x in gallery_law]
	print("Gallery:")
	for x in range(len(gallery)):
		print(f'FP{x+1}: {gallery[x]}')
	#[print(x) for x in gallery]

	# Step 4: Build Score Matrix
	scores = build_score_matrix(probes, gallery)
	print("Score Matrix:")
	print(scores)

	# Step 5: Authentication Process
	threshold = select_threshold(scores)
	auth_matrix, match_pairs = authentication(scores, threshold)
	auth_result(auth_matrix, match_pairs, threshold)

#==============================#
#==========EXTRACTION==========#
#==============================#

# List .tif fingerprint images to confirm they are in directory
dir = os.path.dirname(__file__)
folder = "HW4_FingerprintImages"
path = os.path.join(dir, folder)
tif_files = [f for f in os.listdir(path) if f.endswith('.tif')]

# Load .tif fingerprint images
fingerprint_images = {}

for file_path in tif_files:
    full_file_path = os.path.join(folder, file_path)
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    img = Image.open(full_file_path)
    grayscale_img = img.convert('L')
    fingerprint_images[filename_without_ext] = grayscale_img

# Preprocess images using Gaussian blur to denoise and
# histogram equalization for contrast enhancement
preprocessed_images = {}

for name, pil_img in fingerprint_images.items():
    # Convert PIL Image to NumPy array
    img_np = np.array(pil_img)

    # Apply Gaussian blur for denoising
    img_denoised = cv2.GaussianBlur(img_np, (5, 5), 0)

    # Apply histogram equalization for contrast enhancement
    # Ensure the image type is uint8 for equalizeHist
    img_enhanced = cv2.equalizeHist(img_denoised.astype(np.uint8))

    # Convert NumPy array back to PIL Image
    preprocessed_pil_img = Image.fromarray(img_enhanced)

    preprocessed_images[name] = preprocessed_pil_img

print("Preprocessed images (keys):", preprocessed_images.keys())

# Apply alignment for all images based on users choice of refrence image
# Choose one image to serve as the reference image
reference_name = 'FP3'
reference_pil_img = preprocessed_images[reference_name]
reference_image = np.array(reference_pil_img)

# Create an empty dictionary for aligned images and add the reference
aligned_images = {reference_name: reference_image}

# Iterate through each image for alignment
for name, pil_img in preprocessed_images.items():
    if name == reference_name:
        continue  # Skip the reference image itself

    # Convert PIL Image to NumPy array
    moving_image = np.array(pil_img)

    # Calculate subpixel translation shift
    # The error and diffphase are not needed for this subtask, but phase_cross_correlation returns them
    shift_values, error, diffphase = phase_cross_correlation(reference_image, moving_image, upsample_factor=10)

    # Apply the calculated shift to the moving image
    aligned_img = shift(moving_image, shift_values, mode='constant', cval=0)

    # Store the aligned NumPy array
    aligned_images[name] = aligned_img

# Print the keys of the aligned_images dictionary to confirm
print("\nAligned images (keys):", aligned_images.keys())

# Normalize images before applying Gabor filter
normalized_images = {}

for name, img_np in aligned_images.items():
    # Ensure image is float type for division, then normalize to 0-1 range
    normalized_img = img_np.astype(np.float32) / 255.0
    normalized_images[name] = normalized_img

print("\nNormalized images (keys):", normalized_images.keys())

# Apply Gabor filter to extract fixed-length feature 
# Define frequencies and orientations for the Gabor filters
frequencies = [0.1, 0.3, 0.5] # Example frequencies
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees

# Create an empty dictionary to store the extracted feature vectors
image_features = {}

# Iterate through each normalized image
for name, img_np in normalized_images.items():
    feature_vector = []
    for freq in frequencies:
        for orient in orientations:
            # Apply the Gabor filter
            # gabor_kernel returns the real and imaginary parts of the Gabor response
            # sigma_x and sigma_y are standard deviations of Gaussian envelope
            kernel = gabor_kernel(frequency=freq, theta=orient, sigma_x=4, sigma_y=4)

            # Convolve the image with the Gabor kernel
            # Ensure the image is float for convolution
            real_gabor_response = np.real(np.fft.ifft2(np.fft.fft2(img_np) * np.fft.fft2(kernel, s=img_np.shape)))
            imag_gabor_response = np.imag(np.fft.ifft2(np.fft.fft2(img_np) * np.fft.fft2(kernel, s=img_np.shape)))

            # Calculate the magnitude of the Gabor response
            magnitude = np.sqrt(real_gabor_response**2 + imag_gabor_response**2)

            # Calculate a statistical measure (mean) of the magnitude response
            feature_vector.append(np.mean(magnitude))

    # Store the complete feature_vector
    image_features[name] = np.array(feature_vector)

# Print the keys of the image_features dictionary to confirm
print("\nExtracted image features (keys):", image_features.keys())

# Print the shape of one of the feature vectors to confirm it's fixed-length
if image_features:
    sample_key = next(iter(image_features))
    #print(f"\nShape of a sample feature vector ({sample_key}):", image_features[sample_key])


'''
This section is not necessary for the project but nice to see visualization of the steps

# Select one sample fingerprint image for visualization
sample_name = 'FP3'

# Retrieve images from different stages
original_img = fingerprint_images[sample_name]
preprocessed_img = preprocessed_images[sample_name]
aligned_img = aligned_images[sample_name]
normalized_img = normalized_images[sample_name]

# Define a few Gabor filter parameters to visualize their responses
# Using the same frequencies and orientations as defined before
# Let's pick 4 representative combinations for visualization
sample_gabor_params = [
    {'freq': frequencies[0], 'orient_deg': np.degrees(orientations[0])}, # Freq 0.1, Orient 0
    {'freq': frequencies[0], 'orient_deg': np.degrees(orientations[1])}, # Freq 0.1, Orient 45
    {'freq': frequencies[1], 'orient_deg': np.degrees(orientations[2])}, # Freq 0.3, Orient 90
    {'freq': frequencies[2], 'orient_deg': np.degrees(orientations[3])}  # Freq 0.5, Orient 135
]

sample_gabor_responses = []
for params in sample_gabor_params:
    freq = params['freq']
    orient_rad = np.radians(params['orient_deg'])

    kernel = gabor_kernel(frequency=freq, theta=orient_rad, sigma_x=4, sigma_y=4)

    real_gabor_response = np.real(np.fft.ifft2(np.fft.fft2(normalized_img) * np.fft.fft2(kernel, s=normalized_img.shape)))
    imag_gabor_response = np.imag(np.fft.ifft2(np.fft.fft2(normalized_img) * np.fft.fft2(kernel, s=normalized_img.shape)))

    magnitude = np.sqrt(real_gabor_response**2 + imag_gabor_response**2)
    sample_gabor_responses.append(magnitude)

# Create a figure with multiple subplots
num_gabor_plots = len(sample_gabor_params)
num_rows = 2 # For original, preprocessed, aligned, normalized
num_cols = 2 # For original, preprocessed, aligned, normalized

# If we have Gabor plots, adjust the layout
if num_gabor_plots > 0:
    # Determine number of rows needed for Gabor plots
    gabor_rows = (num_gabor_plots + num_cols - 1) // num_cols # Ceiling division
    fig, axes = plt.subplots(num_rows + gabor_rows, num_cols, figsize=(num_cols * 5, (num_rows + gabor_rows) * 4))
else:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))

axes = axes.flatten() # Flatten the axes array for easy iteration

# Display the pipeline images
images_to_display = [
    (original_img, 'Original Image'),
    (preprocessed_img, 'Preprocessed Image (Denoised, Enhanced)'),
    (aligned_img, 'Aligned Image'),
    (normalized_img, 'Normalized Image (0-1)')
]

for i, (img, title) in enumerate(images_to_display):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"{title} ({sample_name})")
    axes[i].axis('off') # Disable axes ticks

# Display the selected Gabor filter magnitude responses
for i, response in enumerate(sample_gabor_responses):
    plot_idx = len(images_to_display) + i
    params = sample_gabor_params[i]
    axes[plot_idx].imshow(response, cmap='gray')
    axes[plot_idx].set_title(f"Gabor Magnitude\nFreq: {params['freq']:.1f}, Orient: {params['orient_deg']:.0f}°")
    axes[plot_idx].axis('off') # Disable axes ticks

# Hide any unused subplots if there are more axes than plots
for i in range(len(images_to_display) + len(sample_gabor_responses), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
'''

# Once the extraction step is over, we call main to do the rest of the process.
main()