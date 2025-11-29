import numpy as np

def select_threshold(score_matrix):
    threshold = np.median(score_matrix) 
    return threshold

def authentication(score_matrix, threshold):
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

