import statistics as stats
import matplotlib.pyplot as plt

# read accuracy stats from each line and remove empty last line
accuracy_file = open("accuracy.txt")
accuracy_stats = accuracy_file.read().split("\n")
accuracy_stats.pop()

# convert strings from file into floats
accuracy_stats = [float(x) for x in accuracy_stats]

print("Mean: ", stats.mean(accuracy_stats))
print("Median: ", stats.median(accuracy_stats))
print("Mode: ", stats.mode(accuracy_stats))

# plot histogram of sample accuracies
plt.hist(accuracy_stats, bins=10)
plt.title("Eigenface Model Accuracy")
plt.xlabel("Sample Accuracy (%)")
plt.ylabel("Frequency")
plt.show()