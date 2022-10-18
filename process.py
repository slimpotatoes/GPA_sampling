import numpy as np


def sine_1d(array_1d_length, g, delta_g, noise):
    """
    Function to generate a sine function in one dimension with frequency 1/g on the first half and 1/(g+delta+_g) on the
    second half. A phase noise (random variable following a normal distribution) is also added to the sine function.
    :param array_1d_length: integer representing the length of the array and must be greater than 1.
    :param g: float indicating the periodicity of the sine function (representing the crystal periodicity in pixels).
     The float number g must be non-zero.
    :param delta_g: float indicating the variation of the sine function (representing the deformation of
     the crystal). The number delta_g cannot be equal to -g.
    :param noise: float representing the magnitude of the phase noise (variance of the normal distribution) and must be
    greater than or equal to 0.
    :return: 1D numpy array sine noisy sine function with the first half representing the undeformed
    state and the second half the deformed state.
    """
    if g == 0 or delta_g == -g or noise < 0 or array_1d_length < 1:
        raise ValueError('One of the input parameter is invalid')
    else:
        x = np.linspace(0, array_1d_length, array_1d_length + 1)
        n = np.random.normal(loc=0.0, scale=noise, size=np.size(x))
        data = np.append(np.sin(2 * np.pi * x[0:int(np.max(x) / 2)] / g + n[0:int(np.max(x) / 2)]),
                         np.sin(2 * np.pi * x[int(np.max(x) / 2):] / (g + delta_g) + n[int(np.max(x) / 2):]))
        return data


def mask_gaussian_1d(data, center, sigma):
    """
    1D Gaussian function used as a mask in Fourier space to isolate a frequency
    :param data: 1D numpy array
    :param center: float indicating the position of the center of the Gaussian
    :param sigma: float indicating the width of the Gaussian and must be non-zero.
    :return: 1D numpy array of a gaussian function of same size as data centered on the :param center: and following
    the Gaussian distribution of width :param sigma:.
    """
    if np.shape(data)[0] < 1 or sigma == 0:
        raise ValueError('One of the input parameter is invalid')
    else:
        array_1d_length = np.shape(data)[0]
        x = np.linspace(0, array_1d_length - 1, array_1d_length)
        mask = gaussian_1d(x, 1, center, sigma)
        return mask


def mask_position_1d(g, array_1d_length):
    """
    Position in pixels of the center of the 1D gaussian mask
    :param g: float indicating the periodicity of the sine function (representing the crystal periodicity in pixels).
     The float number g must be non-zero.
    :param array_1d_length: integer representing the length of the array and must be greater than 1.
    :return: integer representing the center of the 1D gaussian mask in [-size, size].
    """
    if g == 0:
        raise ValueError('The reference periodicity input is incorrect')
    else:
        k = np.floor(2 / g)  # Moire correction when undersampling
        if k % 2 == 0:
            return round(0.5 * array_1d_length + array_1d_length / g - k / 2 * array_1d_length)
        else:
            return round(0.5 * array_1d_length + array_1d_length / g - (k + 1) / 2 * array_1d_length)


def gpa_moire_correction(result, g):
    """
    Correction in pixels to apply in Fourier space on the undersampled frequency
    :param g: float indicating the periodicity of the sine function (in pixels)
    :result: 1D numpy array
    """
    if g == 0:
        raise ValueError('The reference periodicity input is incorrect')
    else:
        k = np.floor(2 / g)  # Moire correction when undersampling
        if k % 2 == 0:
            return result + k / 2
        else:
            return result + (k + 1) / 2


def strain_1d(result, g):
    """
    Relative strain measurement from the GPA extracted phase using an external reference
    :param result: 1d numpy array
    :param g: float number representing the periodicity of the reference (unstrained state) and must be non-zero
    :return: 1D numpy array representing the relative strain
    """
    if g == 0:
        raise ValueError('The reference periodicity input is incorrect')
    else:
        strain = (1 / g - result) / result
        return strain


def error_gpa_1d(result, g, delta_g):
    """
    Mean square error between the simulated GPA results and the expected results
    :param g: float number representing the periodicity of the reference (unstrained state) and must be non-zero)
    :param delta_g: float number representing the variation of the periodicity g in pixel due to strain
    :result: 1D numpy array of the gpa function
    :return: 1D numpy array of the mean square error at each pixel
    """
    y_ref = result[0:int(np.size(result) / 2)] - 1 / g
    y_strain = result[int(np.size(result) / 2):] - 1 / (g + delta_g)
    y = np.append(y_ref, y_strain)
    return np.sqrt(np.square(y))


def error_strain_1d(strain, g, delta_g):
    """
    Mean square error between the simulated strain results from the gpa calculation and the expected strain results
    :param g: float number representing the periodicity of the reference (unstrained state) and must be non-zero)
    :param delta_g: float number representing the variation of the periodicity g in pixel due to strain
    :strain: 1D numpy array representing the calculated strain function at each pixel
    :return: 1D numpy array of the mean square error at each pixel
    """
    error_strain_zero = strain[0:int(np.size(strain) / 2)]
    error_strain_delta_g = strain[int(np.size(strain) / 2):] + (delta_g / g + delta_g)
    y = np.append(error_strain_zero, error_strain_delta_g)
    return np.sqrt(np.square(y))


def gaussian_1d(x, a, center, sigma):
    """
    1D Gaussian function
    :param x: 1D numpy array
    :param a: float representing the amplitude of the 1D Gaussian function
    :param center: float representing the center of the 1D Gaussian function
    :param sigma: float representing the width of the 1D Gaussian function (must be non-zero)
    :return: 1D numpy array representing a Gaussian with the parameters a, center and sigma
    """
    y = a * np.exp(-0.5 * (x - center) ** 2 / (sigma ** 2))
    return y
