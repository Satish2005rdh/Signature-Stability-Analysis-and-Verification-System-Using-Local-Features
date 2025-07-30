import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} not found.")
    img = cv2.resize(img, (300, 150))  # Resize for uniformity
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_bin

def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_similarity(matches, threshold=50):
    good_matches = [m for m in matches if m.distance < threshold]
    match_ratio = len(good_matches) / len(matches) if matches else 0
    return match_ratio, good_matches

def verify_signature(ref_img_path, test_img_path, similarity_threshold=0.4):
    img1 = preprocess_image(ref_img_path)
    img2 = preprocess_image(test_img_path)

    kp1, desc1 = extract_orb_features(img1)
    kp2, desc2 = extract_orb_features(img2)

    if desc1 is None or desc2 is None:
        print("Not enough features detected.")
        return False

    matches = match_descriptors(desc1, desc2)
    similarity, good_matches = calculate_similarity(matches)

    print(f"Match Similarity: {similarity:.2f}")
    result = similarity >= similarity_threshold
    print("Signature Verified" if result else "Forgery Suspected")

    # Visualization
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    plt.imshow(matched_img, cmap='gray')
    plt.title('Feature Matches')
    plt.axis('off')
    plt.show()

    return result

# Example usage
ref_signature = "Sign/Ref.png"
test_signature = "Sign/test.png"
verify_signature(ref_signature, test_signature)
