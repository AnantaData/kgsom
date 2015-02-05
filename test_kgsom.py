__author__ = 'laksheen'

import unittest
import numpy as np
import kgsom_new
import math

class TestKGSOM(unittest.TestCase):

    def setUp(self):
        print 'in setup'
        self.map = kgsom_new.gsomap()
        #self.map._viewmap()

        data = np.loadtxt("zoo.data.txt",dtype=str,delimiter=",")
        data = np.array(data)
        names = data[:,0]
        names= np.column_stack((names,data[:,-1]))
        self.features= data[:,:-1]
        #print features
        self.features = self.features[:,1:].astype(int)

    def tearDown(self):
        print 'in tear down'
        del self.map
        del self.features

    def test__a(self):

        print 'in test 1'
        print 'number of neurons ',len(self.map.map_neurons)
        list= []
        count=0
        for neu in self.map.map_neurons.values():
            nhash = str(neu.x_c)+" "+str(neu.y_c)
            print neu.weight_vs;
            count += 1
            if len(nhash) != 0:
                list.append(True)
            else:
                list.append(False)

        print self.map.map_neurons
        self.assertTrue(np.all(list))
        self.assertEqual(4,count,'4 neurons not generated')

    def test_gaus_kern(self):

        print 'in test 2'
        vec1= self.features[10]
        vec2= self.features[30]
        numerator = math.pow(np.linalg.norm(vec1-vec2),2)
        gaus_kernel = np.exp(-1*(numerator/(2*self.map.sigma2)))

        self.assertEqual(self.map.gaus_kern(vec1,vec2),gaus_kernel)

    def test_dist_gaus_kern(self):

        print 'in test 3'
        vec1= self.features[20]
        vec2= self.features[50]
        numerator = math.pow(np.linalg.norm(vec1-vec2),2)
        dist = -1*np.exp(-1*(numerator/(2*self.map.sigma2)))
        self.assertEqual(self.map.dist_gaus_kern(vec1,vec2),dist)

    def test_bmu_gaus(self):

        print 'in test 4'
        self.map.map_neurons[str(0)+""+str(1)].weight_vs = self.features[63]
        #print 'map ', self.map.map_neurons[str(0)+""+str(1)].weight_vs
        #print 'features[63] ',self.features[63]
        #print self.map.bmu_gaus(self.features[63])
        self.assertTrue(np.alltrue(self.map.bmu_gaus(self.features[63]).weight_vs==self.features[63]))

    def test_parallel_bmu_search(self):

        print 'in test 5'
        self.map.map_neurons[str(1)+""+str(1)].weight_vs = self.features[63]
        #print 'map ', self.map.map_neurons[str(0)+""+str(1)].weight_vs
        #print 'features[63] ',self.features[63]
        #print self.map.bmu_gaus(self.features[63])
        self.assertTrue(np.alltrue(self.map.parallel_search_bmu(self.features[63]).weight_vs==self.features[63]))

    def test_classified_inputs(self):
        print 'in test 6'
        self.assertIsNotNone(self.map.classified_inputs(),'classified inputs method returned an empty list')

    def test_gaussian_error(self):
        print 'in test 7'
        vec1= self.features[52]
        vec2= self.features[62]
        self.assertTrue(self.map.gaussian_error(vec1,vec2) > 0)
        self.assertTrue(self.map.gaussian_error(vec1,vec1) ==0)

if __name__ == '__main__':
    unittest.main()
