# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import numpy as np
import os

def main():
    print(os.path.basename(__file__))
    print("".join(open(__file__, 'r').readlines()[0:3]))

    bin_sizes = [2,10,51,255]
    categories = [1,4,8]
    training_size = 30
    test_size = 10

    training_labels, training_data = read_batch(
        os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py', 'data_batch_1'), 
        categories, training_size)
    test_labels, test_data = read_batch(
        os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py', 'test_batch'), 
        categories, test_size)

    training_gray = to_gray_scale(training_data)
    test_gray = to_gray_scale(test_data)

    for number_of_bins in bin_sizes:
        bins = make_bins(0, 255, number_of_bins)
        
        training_histograms = np.apply_along_axis(make_histogram, -1, training_data, bins)
        test_histograms = np.apply_along_axis(make_histogram, -1, test_data, bins)
        predictions = np.apply_along_axis(make_prediction, -1, test_histograms, training_histograms, training_labels)

        correct = np.sum(predictions==test_labels)
        incorrect = np.sum(predictions!=test_labels)

        print(f"Bins: {number_of_bins}, correct: {correct}, false: {incorrect}, accuracy: {correct/(correct+incorrect)}")

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_batch(batch_file, categories, per_category):
    batch = unpickle(batch_file)
    labels = np.array(batch[b'labels'])
    data = np.array(batch[b'data'], dtype=np.float32)

    return np.concatenate([labels[labels == cat][:per_category] for cat in categories]), \
           np.concatenate([data[labels == cat][:per_category] for cat in categories])

def to_gray_scale(batch):
    return (batch[:,0:1024]+batch[:,1024:2048]+batch[:,2048:]) / 3

def make_bins(min, max, num):
    return np.arange(min, max + max/num, max/num)

def make_histogram(data, bins): 
    return np.histogram(data,bins=bins)[0]

def make_prediction(single_histogram, training_batch, training_labels):
    diff = training_batch - single_histogram
    return training_labels[np.linalg.norm(diff, axis=1).argmin()]

if __name__ == '__main__':
    main()