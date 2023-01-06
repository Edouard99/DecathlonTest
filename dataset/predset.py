from utils import options,utils
import pandas as pd
import os
from dataset import dataset
from dataset import testdataset
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class PredSet():
    def __init__(self):
        self.past_df=None
        self.pred_df=None
        self.bu_feat_df=None
        self.data_knn=None
        self.initialized=False
        self.params={
                        "max_lat":51.050275,
                        "min_lat":41.9543,
                        "max_long":8.7961,
                        "min_long":-4.4364458576632,
                        "region_idr_cat":np.array([2,3,4,6,7,8,30,31,32,
                                                    33,51,52,53,55,64,65,
                                                    66,69,70,71,72,74,75,
                                                    107,115,121,134,162,178]),
                        "zod_idr_cat":np.array([1,3,4,6,10,35,59,72])
                    }
    def initialize(self,opt:options.BaseOptions):
        self.opt=opt
        self.past_df=pd.read_csv(os.path.join(opt.dataroot,'past_data.csv'))
        self.pred_df=pd.read_csv(os.path.join(opt.dataroot,'prediction_data.csv'))
        self.bu_feat_df=pd.read_csv(os.path.join(opt.dataroot,'bu_feat.csv'))
        self.data_knn=dataset.Annual_construction_dataset()
        with open(os.path.join(opt.dataroot,'data_knn.json'), 'r') as f:
            self.data_knn.load_from_json(f)
        self.data_knn.set_data_for_training(False)
        self.knn_params={
            73:[9,np.array([2/3,0,1/3])],
            88:[8,np.array([2/3,0,1/3])],
            117:[6,np.array([1,0,1])],
            127:[4,np.array([2/3,0,1])]    
        }
        self.dic_knn={
            73:None,
            88:None,
            117:None,
            127:None
        }
        for dep in self.knn_params.keys():
            X_train=[]
            Y_train=[]
            for sample in self.data_knn.samples_per_dep[dep]:
                Y_train.append(sample[0])
                X_train.append(sample[1])
            X_train=np.array(X_train)
            Y_train=np.array(Y_train)
            self.dic_knn[dep] = KNeighborsRegressor(n_neighbors=self.knn_params[dep][0],
                                                weights='distance',
                                                n_jobs=-1,
                                                metric=lambda a,
                                                b:utils.custom_distance(a,b,self.knn_params[dep][1])
                                                ).fit(X_train, Y_train)
        self.dataset_to_nn=testdataset.TestDataset(self.params)
        self.dataset_to_nn.extract_data(self.past_df,self.pred_df,self.bu_feat_df,self.dic_knn)
        self.initialized=True
    
    def write_prediction(self):
        self.results_df=self.pred_df.copy()
        for sample in self.dataset_to_nn.samples:
            bu=sample["bu_num"]
            dep=sample["dep"]
            temp_df=self.results_df[(self.results_df["but_num_business_unit"]==bu) & (self.results_df["dpt_num_department"]==dep)].sort_values(by=["day_id"])
            days=temp_df["day_id"].to_numpy()
            for i in range(len(sample["pred"])):
                self.results_df.loc[(self.results_df["but_num_business_unit"]==bu)
                                        & (self.results_df["dpt_num_department"]==dep)
                                        & (self.results_df["day_id"]==days[i]),"results"]=sample["pred"][i]
        self.results_df.to_csv(os.path.join(self.opt.dataroot,'results.csv'))
        print("You will find your results here {}".format(os.path.join(self.opt.dataroot,'results.csv')))
