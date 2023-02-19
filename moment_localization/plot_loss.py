import matplotlib.pyplot as plt
import numpy as np
import csv
train_losses = []
losses=[]
iteration=[]
with open('plot_folder/loss_all.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
    
    for ROW in plotting:
        iteration.append(float(ROW[0]))
        train_losses.append(float(ROW[1]))

epoch = list(range(1,len(iteration)+1))
plt.figure()
plt.plot(epoch,train_losses, color="blue")
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.savefig('train_loss.png')


# with open('plot_folder/train_loss.txt', 'r') as datafile:
#     plotting = csv.reader(datafile)
    
#     for ROW in plotting:
#         losses.append(float(ROW[2]))

# epoch = list(range(1,len(losses)+1))
# plt.figure()
# plt.plot(epoch,losses, color="blue")
# plt.xlabel('epoch')
# plt.ylabel('train_loss')
# plt.savefig('train_loss.png')