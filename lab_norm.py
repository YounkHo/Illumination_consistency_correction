import cv2
import numpy as np
from scipy.optimize import minimize
from skimage import exposure

def adjust_gamma(image, gamma=1.0):
    invgamma = 1 / gamma
    brighter_image = np.power(image / 255.0, invgamma) * 255.0
    return np.clip(brighter_image, 0, 255)

def get_combined_gamma(source_imgs, target_imgs, beta=0.1, initial_gamma=1.0, size=(256, 256)):
    def loss_function(gamma, source_hist, target_hist, beta):
        L = np.arange(256)
        term1 = np.sum((L ** gamma) * source_hist - L * target_hist) ** 2
        term2 = beta * (gamma - 1) ** 2
        return term1 + term2

    combined_source_hist = np.zeros(256)
    combined_target_hist = np.zeros(256)

    for source_img, target_img in zip(source_imgs, target_imgs):
        # Ensure the images are in RGB format
        source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        source_resized = cv2.resize(source_rgb, size, interpolation=cv2.INTER_AREA)
        target_resized = cv2.resize(target_rgb, size, interpolation=cv2.INTER_AREA)

        source_lab = cv2.cvtColor(source_resized, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_resized, cv2.COLOR_RGB2LAB)

        source_L_hist = cv2.calcHist([source_lab], [0], None, [256], [0, 256]).flatten()
        target_L_hist = cv2.calcHist([target_lab], [0], None, [256], [0, 256]).flatten()

        combined_source_hist += source_L_hist
        combined_target_hist += target_L_hist

    iteration = 0
    def callback(gamma):
        nonlocal iteration
        iteration += 1
        loss = loss_function(gamma, combined_source_hist, combined_target_hist, beta)
        print(f"Iteration {iteration}, gamma: {gamma}, loss: {loss}")

    result = minimize(loss_function, [initial_gamma], args=(combined_source_hist, combined_target_hist, beta),
                      bounds=[(0.0, 3.0)], method='L-BFGS-B', callback=callback,
                      options={'ftol': 1e-7, 'gtol': 1e-5, 'maxiter': 100})

    return result.x

def rectified(test_img, target_img, gamma):
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    target_rgb = cv2.resize(target_rgb, (test_rgb.shape[1], test_rgb.shape[0]), interpolation=cv2.INTER_AREA)

    test_lab = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)

    target_L_min = np.min(target_lab[:, :, 0])
    target_L_max = np.max(target_lab[:, :, 0])
    test_lab[:, :, 0] = adjust_gamma(test_lab[:, :, 0], gamma)

    test_lab[:, :, 0] = (test_lab[:, :, 0] - np.min(test_lab[:, :, 0])) / (np.max(test_lab[:, :, 0]) - np.min(test_lab[:, :, 0])) * (target_L_max - target_L_min) + target_L_min

    for i in range(1, 3):
        test_lab[:, :, i] = (test_lab[:, :, i] + 128)
        target_lab[:, :, i] = (target_lab[:, :, i] + 128)

        test_lab[:, :, i] = exposure.match_histograms(test_lab[:, :, i], target_lab[:, :, i])
        test_lab[:, :, i] = test_lab[:, :, i] - 128

    result_rgb = cv2.cvtColor(test_lab, cv2.COLOR_LAB2RGB)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    A_paths = ["source.png"]
    B_paths = ["target.png"]
    C_path = "test.png"
    beta = 0.1

    source_imgs = [cv2.imread(path) for path in A_paths]
    target_imgs = [cv2.imread(path) for path in B_paths]
    test_img = cv2.imread(C_path)

    # Checking if the images are loaded correctly
    if any(img is None for img in source_imgs) or any(img is None for img in target_imgs) or test_img is None:
        print("Error: One of the images could not be loaded. Please check the file paths.")
    else:
        gamma = get_combined_gamma(source_imgs, target_imgs, beta)
        print(f"Optimal gamma: {gamma}")

        rectified_img = rectified(test_img, target_imgs[0], gamma)  # Use the first target for histogram matching.
        cv2.imwrite("match.png", rectified_img)
