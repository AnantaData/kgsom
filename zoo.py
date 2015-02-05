from scipy.ndimage.interpolation import zoom
from xmlrpclib import boolean
import pandas as pd
import time
import numpy as np
from kgsom import gsomap
import matplotlib.pyplot as plt
import refactored_kgsom

data = np.loadtxt("zoo.data.txt",dtype=str,delimiter=",")

data = np.array(data)

names = data[:,0]

names= np.column_stack((names,data[:,-1]))

features= data[:,:-1]
features = features[:,1:].astype(int)


print features.shape
#preprocess###

#for i in range(features.shape[1]):
#    features[:,i]=features[:,i]/np.max(features[:,i])

norms = np.array([np.linalg.norm(x) for x in features])

sig2 = np.var(norms)
print 'var : ',sig2
##############
positions = np.ndarray(shape=(101,2))
#st = time.time()
'''
gmap = gsomap(SP=0.9999,dims=16,nr_s=10,lr_s=0.01,fd=0.999,lrr=0.95,n_jobs=3,sig2=10000,prune=0.8)
gmap.process_batch(features,100)'''

gmap = refactored_kgsom.get_kgsom(features,positions,names,spread_factor=0.9999,dim=16,nr_s=10,lr_s=0.01,boolean=False,fd=0.999,lrr=0.95,n_jobs=3,sig2=10000,prune=0.8,iterations=100)

'''
print len(gmap.map_neurons.keys())
print (" elapsed time : ",(time.time()-st))

for i in range(positions.shape[0]):
    positions [i]= gmap.predict_point(features[i]).astype(int)
    #print positions[i]

names=np.column_stack((names,positions[:,0],positions[:,1]))

#print names

classification=np.array(['mammal','bird','reptile','fish','amphibian','insect','seacreature'])


labels = names[:,0]
#for i in range(labels.shape[0]):
#    labels[i]=classification[int(labels[i])-1]
x = np.array([i for i in range(100)])
y = np.array(gmap.map_sizes)
plt.plot(x,y)
plt.xlabel("iteration")
plt.ylabel("map size")
plt.show()
plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    positions[:, 0], positions[:, 1], marker = 'o', )
for label, x, y in zip(labels, positions[:, 0], positions[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


plt.show()

colors={1:"green", 2:"yellow", 3:"black", 4:"blue", 5:"red", 6:"orange", 7:"gray"}
colorlist=[]
for x in names[:,1]:
    colorlist.append(colors[int(x)])

colorlist=np.array(colorlist)

sizes = [20*2**2 for n in range(len(names[:,1]))]

plt.scatter(positions[:,0],positions[:,1],c=colorlist,s=sizes)
plt.show()
gmap.viewmap()

#print gmap.map_neurons['010'].weight_vs'''
