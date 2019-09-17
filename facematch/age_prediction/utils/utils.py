import numpy as np

CLASSES_NUMBER = 100
MAX_AGE = 100


def build_age_vector(age, deviation):
    """Build AGE vector as a normal probability histogram"""
    # Sample from a normal distribution using numpy's random number generator
    mean, std = age, deviation / 5.0
    bins_number = deviation * 2 + 1
    samples = np.random.normal(mean, std, size=1000000)

    age_vec = np.zeros(shape=(MAX_AGE))

    # Compute a histogram of the sample
    bins = np.linspace(mean - deviation - 1, mean + deviation - 1, bins_number)
    histogram, bins = np.histogram(samples, bins=bins)

    # Get index of mean in histogram and start / end of histogram in AGE vector
    mean_ind = np.where(histogram == np.amax(histogram))[0][0]
    start_ind = mean - mean_ind
    end_ind = start_ind + histogram.shape[0]

    # handle borders of the probability distribution range falling outside the main range [0..100]
    if start_ind < 0:
        histogram = histogram[abs(start_ind) :]

    if end_ind > MAX_AGE:
        histogram = histogram[: (MAX_AGE - (end_ind))]

    end_ind = min(MAX_AGE, end_ind)
    start_ind = max(0, start_ind)
    age_vec[start_ind:end_ind] = histogram

    # Normalize age histogram
    age_vec = age_vec / age_vec.sum()
    return age_vec
