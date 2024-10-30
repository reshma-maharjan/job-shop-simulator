import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde

timelist_noga = [704, 694, 737, 699, 656, 773, 670, 688, 715, 774, 664, 701, 634, 737, 692, 702, 717, 766, 730, 678, 650, 634, 694, 762, 724, 848, 704, 720, 781, 687, 654, 742, 666, 670, 726, 704, 821, 726, 734, 615, 721, 733, 629, 694, 692, 678, 660, 843, 712, 791, 725, 766, 868, 740, 625, 667, 622, 747, 618, 766, 761, 699, 735, 761, 728, 777, 653, 807, 651, 752, 771, 737, 682, 812, 793, 856, 631, 791, 810, 690, 766, 720, 831, 716, 709, 767]
timelist_ga=[776, 586, 686, 702, 796, 728, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 646, 646, 586, 586, 586, 586, 586, 586, 586, 605, 586, 586, 646, 586, 586, 586, 605, 605, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 605, 586, 586, 586, 586, 586, 586, 605, 605, 586, 586, 586, 586, 586, 586, 646, 646, 586, 586, 586, 605, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586, 586]
timelist_mu2=[689, 667, 856, 668, 757, 784, 667, 667, 646, 602, 667, 667, 586, 586, 666, 747, 627, 627, 667, 667, 668, 668, 586, 586, 668, 668, 667, 667, 657, 657, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 667, 617, 667, 667, 667, 667, 586, 586, 610, 610, 646, 576, 586, 689, 602, 602, 620, 620, 610, 622, 588, 588, 646, 646, 602, 602, 602, 602, 602, 602, 586, 646, 602, 602, 646, 646, 602, 602]
'''
plt.hist(timelist_noga, bins=10, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Frequency Histogram of Time List')
plt.grid(True)
plt.show()


'''


# Plotting the histograms
plt.hist(timelist_noga, bins=20, alpha=0.5, label='Processing time without GA')
plt.hist(timelist_ga, bins=20, alpha=0.5, label='Processing time with GA')

# Adding labels and title
plt.xlabel('Time(second)')
plt.ylabel('Frequency')
plt.title('Frequency Comparison of Processing time without GA and with GA')
plt.legend(loc='upper right')

# Displaying the plot
plt.show()




# Calculate the KDE for each dataset
kde_noga = gaussian_kde(timelist_noga)
kde_ga = gaussian_kde(timelist_ga)

# Plotting the PDFs
x = np.linspace(min(timelist_noga + timelist_ga), max(timelist_noga + timelist_ga), 1000)
plt.plot(x, kde_noga(x), label='Processing time without GA')
plt.plot(x, kde_ga(x), label='Processing time with GA')

# Adding labels and title
plt.xlabel('Time(second)')
plt.ylabel('Probability Density')
plt.title('Probability Density Function (PDF) Comparison')
plt.legend()

# Displaying the plot
plt.show()

# Plotting the CDFs
x = np.linspace(min(timelist_noga + timelist_ga), max(timelist_noga + timelist_ga), 1000)
plt.plot(x, np.cumsum(kde_noga(x)) / np.sum(kde_noga(x)), label='Processing time without GA')
plt.plot(x, np.cumsum(kde_ga(x)) / np.sum(kde_ga(x)), label='Processing time with GA')

# Adding labels and title
plt.xlabel('Time(second)')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function Comparison')
plt.legend()

# Displaying the plot
plt.show()


plt.hist(timelist_ga, bins=20, alpha=0.5, label='Revised Mutation')
plt.hist(timelist_mu2, bins=20, alpha=0.5, label='Original Mutation')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Frequencies')
plt.legend(loc='upper right')
plt.show()


# Probability Density Function (PDF) for both arrays
plt.figure(figsize=(10, 6))
kde_ga = gaussian_kde(timelist_ga)
kde_mu2 = gaussian_kde(timelist_mu2)

x = np.linspace(min(min(timelist_ga), min(timelist_mu2)), max(max(timelist_ga), max(timelist_mu2)), 1000)

plt.plot(x, kde_ga(x), linestyle='-', label='Revised Mutation')
plt.plot(x, kde_mu2(x), linestyle='-', label='Original Mutation')

plt.xlabel('Time (seconds)')
plt.ylabel('Probability Density')
plt.title('PDF Comparison for Both Mutaions')
plt.legend()
plt.show()

# Cumulative Distribution Function (CDF) for both arrays
# Plotting the CDFs
kde_mu2 = gaussian_kde(timelist_mu2)
kde_ga = gaussian_kde(timelist_ga)
x = np.linspace(min(timelist_ga + timelist_mu2), max(timelist_noga + timelist_ga), 1000)
cdf_ga = np.cumsum(kde_ga(x)) / np.sum(kde_ga(x))
cdf_mu2= np.cumsum(kde_mu2(x)) / np.sum(kde_mu2(x))
plt.plot(x, cdf_ga, label='Revised Mutation')
plt.plot(x, cdf_mu2, label='Original Mutation')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function Comparison')
plt.legend()

# Saving the plot as a PDF
plt.show()