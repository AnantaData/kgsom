import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy import Infinity, Inf, shape
from scipy.spatial.distance import minkowski, jaccard
from Crypto.Util.number import size
import sys
from joblib import Parallel, delayed


### Helper methods ####
def _find_in_map(gmap,ix_rng_s,ix_rng_e, inp_vec):

    keys= gmap.keys()[ix_rng_s:ix_rng_e]

    minDist=9223372036854775807
    candidate= None
    for neu_key in keys:
        neu = gmap[neu_key]
        cand=minkowski(inp_vec, neu.weight_vs, 2)
        if minDist> cand:
            minDist = cand
            candidate= neu

    return  candidate
#####

class neuron(object):

    x_c = 0
    y_c = 0
    res_err = 0
    weight_vs=None
    #new entry
    coassoc_vs = None

    def __init__(self,x,y,dims):
        """

        :rtype : object
        """
        self.time = 0
        self.hits=0
        self.x_c=x
        self.y_c=y
        self.weight_vs=np.random.random(size=dims)
        self.coassoc_vs=np.zeros(shape=(100000))

    def coords(self):
        self.cors= [str(self.x_c), str(self.y_c)]
        return np.array(self.cors)


class gsomap(object):

    map_neurons ={}
    t_time = 0
    count = 0

    def __init__(self,SP=0.5,dims=3,nr_s=6,lr_s=0.9,boolean=False, lrr =0.5,fd=0.5, n_jobs = 2,sig2=0.2, prune=0.5):
        self.prun_coef=prune
        self.sigma2=sig2
        self.n_jobs= n_jobs
        self.boolean=boolean
        self.fd=fd
        self.lr_red_coef = lrr
        self.dim = dims
        self.t_time = 2
        self.map_sizes = []
        for i in range(4):
            x=i/2
            y=i%2
            nhash = str(x)+""+str(y)
            self.map_neurons[nhash] = neuron(i/2, i%2, dims)
            n = self.map_neurons[nhash]
            n.time=1
            self.map_neurons[nhash]=n

        ''' definition of growth threshold according to GSOM paper'''

        self.thresh=-1*dims*np.log(SP)
        print "error threshold : ",self.thresh
        self.nr=nr_s
        self.lr=lr_s
        #print self.map_neurons


    def _gaus_kern(self, u, v):
        return np.exp(-1*(np.linalg.norm((u-v))*(np.linalg.norm(u-v))/(2*self.sigma2)))


    def _dist_gaus_kern(self, v, u):

        return -1*self._gaus_kern(u,v)

    def _adjustment_gaus(self, x, m):
        return self._gaus_kern(x,m)*(x-m)

    def bmu_gaus(self,input_vec):
        minDist=9223372036854775807
        candidate= None
        for neu in self.map_neurons.itervalues():
            #print "input: "+str(input_nparray)
            #print "neuron: "+str (neu.weight_vs)

            cand=self._dist_gaus_kern(input_vec, neu.weight_vs)
            if minDist> cand:
            #print "mindist:",minDist
            #print "cand:",cand
                minDist = cand
                candidate= neu

                #print "candidate'scoords",candidate.coords()
        return  candidate


    def _parallel_search_bmu(self, input_vector):

        mapsize=len(self.map_neurons.keys())
        indices=[]
        for i in range (self.n_jobs):
            indices.append([i*mapsize/self.n_jobs , (i+1)*mapsize/self.n_jobs-1])

        #res = Parallel(n_jobs=2) (delayed(check_paths) (Path(points), a) for points in b)

        res=Parallel(n_jobs=self.n_jobs)(delayed(_find_in_map)(self.map_neurons,ix_range[0],ix_range[1],input_vector)for ix_range in indices)
        for r in res:
            if r is None:
                print r

        minDist=9223372036854775807
        candidate= None
        for neu in res:
            cand=minkowski(input_vector, neu.weight_vs, 2)
            if minDist> cand:
                minDist = cand
                candidate= neu

        return candidate


    def _viewmap(self):
        x=np.ndarray(shape=len(self.map_neurons))
        y=np.ndarray(shape=len(self.map_neurons))
        i=0;
        for neu in self.map_neurons.itervalues():
            x[i]=neu.x_c
            y[i]=neu.y_c
            i+=1

        plt.plot_date(x, y)
        plt.xlabel("xcoords")
        plt.ylabel("ycoords")
        plt.show()


    def _predict_point(self, input_array):
        bmu = self._getBMU(input_array)
        #bmu = self._parallel_search_bmu(input_array)
        return bmu.coords()


    def _jaccard_sim(self,nparray1, nparray2,sym=True):

        b1=nparray1.astype(bool)
        b2=nparray2.astype(bool)
        if sym:
            out = len(np.where(np.logical_and(b1,b2))[0])+len(np.where(np.logical_or(b1,b2)==False)[0])
        else:
            out= len(np.where(np.logical_and(b1,b2))[0])

        return out/len(b1)

    def _gaussian_error(self, u, v):
        errsq = self._gaus_kern(u,u)+self._gaus_kern(v,v)-2*self._gaus_kern(u,v)
        err = np.sqrt(errsq)
        return err

    def _process_input(self,input_np_array):
        bmu = self._getBMU(input_np_array)
        bmu.hits += 1
        bmu.time = self.count
        for neu in self.map_neurons.values():
            nhash = str(neu.x_c)+""+str(neu.y_c)
           # print "bmu: "+str(bmu.coords())
            #print "neu: "+str(neu.coords())nhash
            dist =  minkowski(bmu.coords().astype(float), neu.coords().astype(float), 2)
            if dist< self.nr:
                '''weight adjustment *np.exp(-1*dist**2/2*self.nr**2)'''
                #neu.weight_vs = neu.weight_vs + self.lr * (input_np_array-neu.weight_vs)
                neu.weight_vs = neu.weight_vs + self.lr *np.exp(((dist/self.nr)**2)/(-2))* self._adjustment_gaus(input_np_array,neu.weight_vs)
                err =self._gaussian_error(input_np_array,neu.weight_vs)
                neu.res_err += err#minkowski(neu.weight_vs, bmu.weight_vs, 2)
                self.map_neurons[nhash]=neu

        if bmu.res_err > self.thresh:
            #print bmu.res_err
            neu = bmu
            down=str(neu.x_c)+str(int(neu.y_c)-1)
            up=str(neu.x_c)+str(int(neu.y_c)+1)
            left=str(int(neu.x_c)-1)+str(neu.y_c)
            right=str(int(neu.x_c)+1)+str(neu.y_c)
            nei_coords = np.array([down, up , left , right ] )
            nei_coordi = np.array([[(neu.x_c),(int(neu.y_c)-1)], [(neu.x_c),(int(neu.y_c)+1)], [(int(neu.x_c)-1),(neu.y_c)], [(int(neu.x_c)+1),int(neu.y_c)]] )
            p =0
            for coord in nei_coords:
                n=None
                try:
                    n= self.map_neurons[coord]
                    n.res_err+=self.fd*n.res_err

                except KeyError:
                    nwron=neuron(nei_coordi[p][0], nei_coordi[p][1], self.dim)
                    nwron.time=self.t_time
                    new_weight = 0
                    #case a) new node has two consecutive nodes on one of its sides
                    #tiroshan and lakmal please implement the code here
                    #case b) between two old nodes
                    new_weight_b = self._type_b_weight_init(p,neu)
                    new_weight_a = self._type_a_weight_init(p,neu)
                    new_weight_c = self._type_c_weight_init(p,neu)

                    if new_weight_b.all() ==0:
                        if new_weight_a.all() == 0:
                            if new_weight_c.all() == 0:
                                #print "c==0"
                                new_weight = np.ndarray(shape=(self.dim))
                                new_weight.fill(0.5)
                            else:
                                new_weight = new_weight_c
                        else:
                            new_weight = new_weight_a
                    else:
                        new_weight = new_weight_b

                    # nwron.weight_vs=np.ndarray(shape=(self.dim))
                    # nwron.weight_vs.fill(0.5)
                    nwron.weight_vs = new_weight
                    n=nwron
                self.map_neurons[coord]=n
                p+=1
            bmu.res_err=self.thresh/2
            self.map_neurons[str(bmu.x_c)+""+str(bmu.y_c)]=bmu
        return bmu.coords()


    def _getBMU(self,input_nparray):
        minDist=9223372036854775807
        candidate= None
        for neu in self.map_neurons.itervalues():
            #print "input: "+str(input_nparray)
            #print "neuron: "+str (neu.weight_vs)
            if self.boolean:
                cand = jaccard(input_nparray, neu.weight_vs)
                if minDist> cand:
                    minDist = cand
                    candidate= neu
            else:
                cand=minkowski(input_nparray, neu.weight_vs, 2)
                if minDist> cand:
                #print "mindist:",minDist
                #print "cand:",cand
                    minDist = cand
                    candidate= neu

                #print "candidate'scoords",candidate.coords()
        return  candidate

    def _type_b_weight_init(self, side, neuron):
        if side == 0:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c-2))
        elif side == 1:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c+2))
        elif side == 2:
            next_watch = str(int(neuron.x_c-2))+str(int(neuron.y_c))
        else :
            next_watch = str(int(neuron.x_c+2))+str(int(neuron.y_c))

        try:
            b_type_nwron = self.map_neurons[next_watch]
            new_weight = (b_type_nwron.weight_vs+neuron.weight_vs)/2
        except:
            #new_weight = np.ndarray(shape=(self.dim))
            #new_weight.fill(0)
            new_weight = np.array([0])
        return new_weight

    def _type_a_weight_init(self, side, neuron):
        if side == 0:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c+1))
        elif side == 1:
            next_watch = str(int(neuron.x_c))+str(int(neuron.y_c-1))
        elif side == 2:
            next_watch = str(int(neuron.x_c+1))+str(int(neuron.y_c))
        else :
            next_watch = str(int(neuron.x_c-1))+str(int(neuron.y_c))

        try:
            a_type_nwron = self.map_neurons[next_watch]
            if np.linalg.norm(a_type_nwron.weight_vs) > np.linalg.norm(neuron.weight_vs):
                new_weight = neuron.weight_vs - (a_type_nwron.weight_vs - neuron.weight_vs)
            else:
                new_weight = neuron.weight_vs + (neuron.weight_vs - a_type_nwron.weight_vs)
        except:
            #new_weight = np.ndarray(shape=(self.dim))
            #new_weight.fill(0)
            new_weight =np.array([0])
        return new_weight

    def _type_c_weight_init(self, side, neuron):
        if side == 0 or side == 1:
            next_watch_1 = str(int(neuron.x_c+1))+str(int(neuron.y_c))
            next_watch_2 = str(int(neuron.x_c-1))+str(int(neuron.y_c))
        else:
            next_watch_1 = str(int(neuron.x_c))+str(int(neuron.y_c+1))
            next_watch_2 = str(int(neuron.x_c))+str(int(neuron.y_c-1))

        try:
            c_type_nwron = self.map_neurons[next_watch_1]
            if np.linalg.norm(c_type_nwron.weight_vs) > np.linalg.norm(neuron.weight_vs):
                new_weight = neuron.weight_vs - (c_type_nwron.weight_vs - neuron.weight_vs)
            else:
                new_weight = neuron.weight_vs + (neuron.weight_vs - c_type_nwron.weight_vs)
        except:
            try:
                c_type_nwron = self.map_neurons[next_watch_2]
                if np.linalg.norm(c_type_nwron.weight_vs) > np.linalg.norm(neuron.weight_vs):
                    new_weight = neuron.weight_vs - (c_type_nwron.weight_vs - neuron.weight_vs)
                else:
                    new_weight = neuron.weight_vs + (neuron.weight_vs - c_type_nwron.weight_vs)
            except:
                #new_weight = np.ndarray(shape=(self.dim))
                #new_weight.fill(0)
                new_weight=np.array([0])

        return new_weight

    def _grow_map(self,input,i):
        bcoords=self._process_input(input)
        bhash=str(bcoords[0])+""+str(bcoords[1])
        winner = self.map_neurons[bhash]

        score= minkowski(winner.weight_vs,input,2)
        winner.coassoc_vs[i]= score

        self.map_neurons[bhash]=winner
        return

def get_kgsom(data,positions,names,spread_factor,dim,nr_s,lr_s,boolean,fd,lrr,n_jobs,sig2,prune,iterations=10):

    st = time.time()
    kgsom = gsomap(spread_factor,dim,nr_s,lr_s,boolean,fd,lrr,sig2,prune)
    start_time= time.time()
    kgsom.count = 1
    for j in range(iterations):
        kgsom.t_time += 1
        kgsom.map_sizes.append(len(kgsom.map_neurons.values()))
        for i in range(data.shape[0]):
            kgsom.count += 1
            sys.stdout.write("iteration %d :"%(j+1))
            sys.stdout.write(" : NR = %d: "%(kgsom.nr))
            sys.stdout.write(" input %d "%(i))
            sys.stdout.write(" map size %d "%(len(kgsom.map_neurons.keys())))
            sys.stdout.write(" time %d \r"%(time.time()-start_time))
            sys.stdout.flush()
            input = data[i]
            kgsom._grow_map(input,i)

        kgsom.nr=kgsom.nr*(1-kgsom.lr)
        kgsom.lr = kgsom.lr*kgsom.lr_red_coef*(1-3.85/len(kgsom.map_neurons.values()))
        if kgsom.nr <=1 :
            print kgsom.nr
            return
        for l in kgsom.map_neurons.keys():
            if kgsom.map_neurons[l].hits <= kgsom.prun_coef*(0.1**iterations)*kgsom.count*(9**(j)) or kgsom.map_neurons[l].time <= j*kgsom.t_time/iterations:
                del kgsom.map_neurons[l]

    print len(kgsom.map_neurons.keys())
    print (" elapsed time : ",(time.time()-st))

    for i in range(positions.shape[0]):
        positions [i]= kgsom._predict_point(data[i]).astype(int)
    #print positions[i]

    names=np.column_stack((names,positions[:,0],positions[:,1]))

#print names

    classification=np.array(['mammal','bird','reptile','fish','amphibian','insect','seacreature'])

    labels = names[:,0]
#for i in range(labels.shape[0]):
#    labels[i]=classification[int(labels[i])-1]
    x = np.array([i for i in range(100)])
    y = np.array(kgsom.map_sizes)
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
    kgsom._viewmap()

    return
