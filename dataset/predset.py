from utils import options,utils
import pandas as pd
import os
from dataset import dataset
from dataset import testdataset
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import torch

class PredSet():
    """
    Prediction set used for prediction.
    The minimum date of prediction is the minimum date in pred_df.

    Attributes:
        -past_df: a dataframe containing columns : ["day_id],["but_num_business_unit"],
            ["dpt_num_department"],["turnover"]. This dataframe must contains at least 16weeks
            of data before the minimum date of prediction (from minimumdate-16weeks to minimumdate-1weeks) for all business 
            unit and department id in pred_df.
        -pred_df: a dataframe containing columns : ["day_id],["but_num_business_unit"],
            ["dpt_num_department"]. This dataframe must contains 7 weeks of data after minimum prediction date (from minimumdate
            to minimumdate+7weeks) for all business unit and department id for which the prediction will be done.
        -bu_feat_df: a dataframe containing columns : ["but_num_business_unit"],["but_latitude"],
            ["but_longitude"],["but_region_idr_region"],["zod_idr_zone_dgr"], it must contains data for all business 
            unit and department id in pred_df.
        -data_knn(dataset.Annual_construction_dataset): a dataset that is loaded from data_knn.json file, that contains data
            to fit the k nearest neighbors models.
        -initialized(bool): True if the predset has been initialized with initialize()
        -params(dic): params used for normalization and one-hot encoding to build TestDataset.
        -opt(utils.options.BaseOptions): Namespace that contains options
        -knn_params(dic): knn hyperparameters per department.
        -dic_knn(dic): a dictionnary that contains fitted k nearest neighbors models for each department
        -dataset_to_nn(dataset.testdataset.TestDataset): dataset that contains the data preprocessed to feed the neural network model.
        -results_df: a dataframe containing columns : ["day_id],["but_num_business_unit"],
            ["dpt_num_department"],["results"]. The ["results"] contains the predictions of the network.
    Methods:
        initialize : load the dataframes, fit the nearest neighbors models and build a dataset that contains preprocessed data to
            feed the neural network for prediction.
        write_prediction : write the predictions of the network to a dataframe and a csv.
    """
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
    def initialize(self,opt):
        """
        Loads the dataframes, fit the nearest neighbors models and build a dataset that contains preprocessed data to
        feed the neural network for prediction.
        Args:
            -opt : a Namespace containing dataroot.
        """
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
        """
        Writes the predictions of the network to a results_df and to dataroot/results.csv.
        """
        self.results_df=self.pred_df.copy()
        for sample in self.dataset_to_nn.samples:
            bu=sample["bu_num"]
            dep=sample["dep"]
            if (bu==self.opt.bu and dep==self.opt.dep and self.opt.plot==1):
                fig=plt.figure(figsize=(10,5))
                plt.plot(np.concatenate([sample["x"],sample["pred"]]),label="Prediction",color="red")
                plt.plot(sample["x"],label="Turnover until now",color="blue")
                plt.plot(utils.inversenorm(np.concatenate([sample["annual_x"].numpy(),sample["annual_y"].numpy()]),sample["max_turn"],sample["min_turn"]),color="green",label="Annual Average")
                plt.title("Turnover Prediction")
                plt.legend()
                plt.show()
            temp_df=self.results_df[(self.results_df["but_num_business_unit"]==bu) & (self.results_df["dpt_num_department"]==dep)].sort_values(by=["day_id"])
            days=temp_df["day_id"].to_numpy()
            for i in range(len(sample["pred"])):
                self.results_df.loc[(self.results_df["but_num_business_unit"]==bu)
                                        & (self.results_df["dpt_num_department"]==dep)
                                        & (self.results_df["day_id"]==days[i]),"turnover"]=sample["pred"][i]
        self.results_df.to_csv(os.path.join(self.opt.dataroot,'results.csv'),index=False)
        print("You will find your results here {}".format(os.path.join(self.opt.dataroot,'results.csv')))
