import datastruct as datastruct
import process as process
import gpa as gpa

# Example of application (Test #1 from jupyter notebook)

length = 1023
freq = 4.0
center = 768
sigma = 10
strain = 0
noise = 0

data = datastruct.GPAdata()
data.sine = process.sine_1d(length, freq, strain , noise)
data.mask = process.mask_gaussian_1d(data.sine, center, sigma)
data.gpa = gpa.gpa(data.sine, data.mask)
data.strain = process.strain_1d(data.gpa, freq)
data.error_gpa = process.error_gpa_1d(data.gpa, freq, strain)