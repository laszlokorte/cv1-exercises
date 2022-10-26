import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def to_gray_scale(batch):
    return (batch[:,0:1024]+batch[:,1024:2048]+batch[:,2048:]) / 3

def generate_bins(min, max, num):
    return np.arange(min, max + max/num, max/num)

def histogram(data,bins): 
    return np.histogram(data,bins=bins)[0]

if __name__ == '__main__':
    training_batch_file = './cifar-10-batches-py/data_batch_1'
    test_batch_file = './cifar-10-batches-py/test_batch'

    training_batch = unpickle(training_batch_file)
    test_batch = unpickle(test_batch_file)

    bins = generate_bins(0, 255, 51)

    training_labels_all = np.array(training_batch[b'labels'])
    training_data_all = np.array(training_batch[b'data'])

    training_labels_automobile = training_labels_all[training_labels_all==1][:30]
    training_labels_deer = training_labels_all[training_labels_all==4][:30]
    training_labels_ship = training_labels_all[training_labels_all==8][:30]

    training_automobile = training_data_all[training_labels_all==1][:30]
    training_deer = training_data_all[training_labels_all==4][:30]
    training_ship = training_data_all[training_labels_all==8][:30]

    training_automobile_gray = to_gray_scale(training_automobile)
    training_deer_gray = to_gray_scale(training_deer)
    training_ship_gray = to_gray_scale(training_ship)


    training_automobile_hist = np.apply_along_axis(histogram, -1, training_automobile_gray, bins)
    training_deer_hist = np.apply_along_axis(histogram, -1, training_deer_gray, bins)
    training_ship_hist = np.apply_along_axis(histogram, -1, training_ship_gray, bins)
    
    training_data = np.concatenate((training_automobile_hist, training_deer_hist, training_ship_hist))
    training_labels = np.concatenate((training_labels_automobile, training_labels_deer, training_labels_ship))


    test_labels_all = np.array(test_batch[b'labels'])
    test_data_all = np.array(test_batch[b'data'])

    test_labels_automobile = test_labels_all[test_labels_all==1][:30]
    test_labels_deer = test_labels_all[test_labels_all==4][:30]
    test_labels_ship = test_labels_all[test_labels_all==8][:30]

    test_automobile = test_data_all[test_labels_all==1][:30]
    test_deer = test_data_all[test_labels_all==4][:30]
    test_ship = test_data_all[test_labels_all==8][:30]

    test_automobile_gray = to_gray_scale(test_automobile)
    test_deer_gray = to_gray_scale(test_deer)
    test_ship_gray = to_gray_scale(test_ship)

    test_automobile_hist = np.apply_along_axis(histogram, -1, test_automobile_gray, bins)
    test_deer_hist = np.apply_along_axis(histogram, -1, test_deer_gray, bins)
    test_ship_hist = np.apply_along_axis(histogram, -1, test_ship_gray, bins)
    
    test_data = np.concatenate((test_automobile_hist, test_deer_hist, test_ship_hist))
    test_labels = np.concatenate((test_labels_automobile, test_labels_deer, test_labels_ship))

    # plt.imshow(np.concatenate((test_deer[10,:1024].reshape(32,32,1), test_deer[10,1024:2048].reshape(32,32,1), test_deer[10,2048:].reshape(32,32,1) ), -1))
    # plt.show()
    # plt.imshow(np.concatenate((test_automobile[10,:1024].reshape(32,32,1), test_automobile[10,1024:2048].reshape(32,32,1), test_automobile[10,2048:].reshape(32,32,1) ), -1))
    # plt.show()
    # plt.imshow(np.concatenate((test_ship[10,:1024].reshape(32,32,1), test_ship[10,1024:2048].reshape(32,32,1), test_ship[10,2048:].reshape(32,32,1) ), -1))
    # plt.show()

    count_true = 0
    count_false = 0
    for real_label, test_d in zip(test_labels, test_data):
        min_distance = float('inf')
        prediction = None
        for training_label, training_d in zip(training_labels, training_data):
            dist = np.linalg.norm(training_d - test_d)
            if dist < min_distance:
                min_distance = dist
                prediction = training_label
        if prediction == real_label:
            count_true += 1
        else:
            count_false += 1

    print(f"correct: {count_true}, false: {count_false}")
    #deer_hist = np.apply_along_axis(np.histogram, -1, deer_gray, bins)
    #ship_hist = np.apply_along_axis(np.histogram, -1, ship_gray, bins)
    # print(automobile_hist)
    # print(batch[b'batch_label'])
    # print(batch[b'labels'])
    # print(batch[b'data'])