import os
import librosa
import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler


def extract_mfcc(file_path, n_mfcc=20):
    """
    Extract MFCC features from an audio file.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCCs to extract.

    Returns:
        np.ndarray: MFCC feature matrix.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Normalize MFCC
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def compute_statistics(mfcc_features):
    """
    Compute the mean and covariance of MFCC features.

    Args:
        mfcc_features (list of np.ndarray): List of MFCC feature matrices.

    Returns:
        tuple: Mean vector and covariance matrix.
    """
    # Concatenate all MFCC features
    all_features = np.hstack(mfcc_features)
    mean = np.mean(all_features, axis=1)
    cov = np.cov(all_features)
    return mean, cov


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet Distance between two Gaussians.

    Args:
        mu1 (np.ndarray): Mean vector of the first distribution.
        sigma1 (np.ndarray): Covariance matrix of the first distribution.
        mu2 (np.ndarray): Mean vector of the second distribution.
        sigma2 (np.ndarray): Covariance matrix of the second distribution.
        eps (float): Small value to add to the diagonal of covariance matrices for numerical stability.

    Returns:
        float: The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("Adding epsilon to diagonal of covariance matrices for numerical stability.")
        offset = eps * np.eye(sigma1.shape[0])
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fd


def load_features(folder_path, file_extension):
    """
    Load and extract MFCC features from all audio files in a folder.

    Args:
        folder_path (str): Path to the folder containing audio files.
        file_extension (str): Extension of the audio files (e.g., '.wav', '.mp3').

    Returns:
        list of np.ndarray: List of MFCC feature matrices.
    """
    mfcc_features = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_extension):
            file_path = os.path.join(folder_path, filename)
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                mfcc_features.append(mfcc)
    return mfcc_features


def main(generated_folder, ground_truth_folder):
    # Load and extract MFCC features
    print("Extracting MFCC features from ground truth...")
    ground_truth_features = load_features(ground_truth_folder, '.mp3')
    if not ground_truth_features:
        print("No ground truth features extracted. Exiting.")
        return

    print("Extracting MFCC features from generated audio...")
    generated_features = load_features(generated_folder, '.wav')
    if not generated_features:
        print("No generated features extracted. Exiting.")
        return

    # Compute statistics
    print("Computing statistics for ground truth...")
    mu_gt, sigma_gt = compute_statistics(ground_truth_features)

    print("Computing statistics for generated audio...")
    mu_gen, sigma_gen = compute_statistics(generated_features)

    # Compute Frechet Distance
    print("Calculating Frechet Distance...")
    fd = frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)

    print(f"Frechet Distance between generated audio and ground truth: {fd:.4f}")

#Frechet Distance between generated audio and ground truth: 1.0153 '/Users/mac/Downloads/maplestory background music'
#Frechet Distance between generated audio and ground truth: 0.7420 '/Users/mac/Downloads/upbeat orchestral piece with jazz influences, featuring piano and strings, capturing the whimsical and adventurous spirit of a fantasy world'
#Frechet Distance between generated audio and ground truth: 0.6013 '/Users/mac/Downloads/electronic track with a lighthearted and playful mood, incorporating synthesizers and woodwind instruments, suitable for a vibrant game environment.'


if __name__ == "__main__":
    generated_folder = '/Users/mac/Downloads/electronic track with a lighthearted and playful mood, incorporating synthesizers and woodwind instruments, suitable for a vibrant game environment.'
    ground_truth_folder = '/Users/mac/PycharmProjects/ALL_Shit/my_Stuff/maplestory_music/musicgen/data_10min_chunks/original_maple'
    main(generated_folder, ground_truth_folder)