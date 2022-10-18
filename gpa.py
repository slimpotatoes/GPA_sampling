import numpy as np
from skimage.restoration import unwrap_phase


def gpa(data, mask):
    """
    Geometrical Phase Analysis function to obtain the frequency of a periodic function and its local variation
    :param data: N dimension np.array
    :param mask: N dimension np.array of same shape as data masking a section of the data in the N dimension Fourier
    space
    :return: N*N dimensions np.array showing the distribution of the phase variations of the isolated frequency over the
    N*N dimensions
    """
    x = np.multiply(mask, np.fft.fftshift(np.fft.fft(data)))
    phase = unwrap_phase(np.angle(np.fft.ifft(np.fft.ifftshift(x))))
    return 1 / (2 * np.pi) * np.gradient(phase)
