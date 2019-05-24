import numpy as np
from collections import defaultdict
from DDC import DDCSSM, History

class MeanLearner(DDCSSM):
    
    def online_initialise(self, train_data, test_data):

        self.train_history = History()
        self.test_history = History()
        
        # training data
        self.train_data = train_data.copy()     
        self.test_data = test_data.copy()     
        
        self.train_data["mzt-1_x1:t-1"]  = self.ddc.transform_data(train_data, ["zt-1"])["zt-1"]
        self.train_data["mzt_x1:t"]  = self.ddc.transform_data(train_data, ["zt"])["zt"]
        self.train_data["mxt_x1:t-1"]  = self.ddc.transform_data(train_data, ["xt"])["xt"]

        self.W_trans = self.ddc.train_weights(train_data, "zt-1->zt")
        self.W_emit  = self.ddc.train_weights(train_data, "zt->xt")

        self.test_data["mzt-1_x1:t-1"]  = self.train_data["mzt-1_x1:t-1"]
        self.test_data["mzt_x1:t"]      = self.train_data["mzt_x1:t"]
        
    def online_propagate(self, test_data):

        self.train_data["mzt-2_x1:t-2"] = self.train_data["mzt-1_x1:t-1"].copy()
        self.train_data["mzt-1_x1:t-1"] = self.train_data["mzt_x1:t"].copy()
        self.train_data[  "mzt_x1:t-1"] = self.train_data["mzt-1_x1:t-1"].dot(self.W_trans)

        self.train_data["mxt-1_x1:t-2"] = self.train_data["mxt_x1:t-1"].copy()
        self.train_data[  "mxt_x1:t-1"] = self.train_data["mzt_x1:t-1"].dot(self.W_emit)

        self.test_data[ "mzt-2_x1:t-2"] = self.test_data[ "mzt-1_x1:t-1"].copy()
        self.test_data[ "mzt-1_x1:t-1"] = self.test_data[ "mzt_x1:t"].copy()

        self.test_data["xt-1"] = self.test_data["xt"]
        
        for k, v in test_data.items():
            self.test_data[k]=v.copy()
            
class BilinearLearner(MeanLearner):
    
    name="bilinear"
    
    def __init__(self, fs, lam=1e-5):
        super(BilinearLearner, self).__init__(fs, lam=lam)
        self.name="bilinear"

    
    def online_train_test(self):

        self.ddc.train_weights(self.train_data, "mzt-2_x1:t-2,mxt-1_x1:t-2->mzt-1_x1:t-1")
        self.ddc.train_weights(self.train_data, "mzt-2_x1:t-2,mxt-1_x1:t-2->mzt_x1:t-1")
        self.ddc.train_weights(self.train_data, "mzt-1_x1:t-1,mxt_x1:t-1->mzt_x1:t")

        self.ddc.copy_weights("mzt-2_x1:t-2,mxt-1_x1:t-2->mzt-1_x1:t-1", "mzt-2_x1:t-2,xt-1->mzt-1_x1:t-1")
        self.ddc.copy_weights("mzt-2_x1:t-2,mxt-1_x1:t-2->mzt_x1:t-1",   "mzt-2_x1:t-2,xt-1->mzt_x1:t-1")
        self.ddc.copy_weights("mzt-1_x1:t-1,mxt_x1:t-1->mzt_x1:t",       "mzt-1_x1:t-1,xt->mzt_x1:t")
            
        self.test_data["mzt-1_x1:t-1"] = self.ddc.predict(self.test_data, "mzt-2_x1:t-2,xt-1->mzt-1_x1:t-1")
        self.test_data[  "mzt_x1:t-1"] = self.ddc.predict(self.test_data, "mzt-2_x1:t-2,xt-1->mzt_x1:t-1")
        self.test_data[    "mzt_x1:t"] = self.ddc.predict(self.test_data, "mzt-1_x1:t-1,xt->mzt_x1:t")

        self.W_trans = self.ddc.train_weights(self.test_data, "mzt_x1:t-1->mzt_x1:t-1")
        self.W_emit  = self.ddc.train_weights(self.test_data, "mzt_x1:t-1->xt")
       
    def online_test(self):
        
        self.test_data[ "mzt_x2:t"] = self.ddc.predict(self.test_data,  "mzt-1_x1:t-1,xt->mzt_x1:t")   
