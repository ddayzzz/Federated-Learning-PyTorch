import sys
import os
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        exit(f'{__file__} <losses_dir>')
    prefix = sys.argv[1]
    loss_objs = dict()
    dc_objs = dict()
    for target_name in os.listdir(prefix):
        with open(os.sep.join((prefix, target_name)), 'rb') as f:
            obj = pickle.load(f)
            loss_objs[target_name], dc_objs[target_name] = obj
    plt.figure()
    plt.title('Avg Training Loss vs Communication rounds')
    plt.ylabel('Avg Training loss')
    plt.xlabel('Communication Rounds')
    for target_name, loss in loss_objs.items():
        ax = plt.plot(range(len(loss)), loss, label=target_name)
    plt.legend()
    plt.show()
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Avg DC vs Communication rounds')
    plt.ylabel('Avg DC')
    plt.xlabel('Communication Rounds')
    for target_name, loss in dc_objs.items():
        ax = plt.plot(range(len(loss)), loss, label=target_name)
    plt.legend()
    plt.show()