

# class BaseOptions():
#     def __init__(self):
#         self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#         self.initialized = False

#     def initialize(self):
#         self.parser.add_argument('--dataroot', type=str, required=True,
#                                  help='path to folder (should contains past_data.csv, prediction_data.csv, bu_feat.csv, data_knn.json)')
#         self.parser.add_argument('--device', type=int, default=0,
#                                  help='choose between cpu=0(default) and gpu=1')
#         self.parser.add_argument('--bu', type=int, default=95,
#                                  help='business unit number for which you will plot the prediction')
#         self.parser.add_argument('--dep', type=int, default=73,
#                                  help='department number for which you will plot the prediction')
#         self.parser.add_argument('plot', type=int, default=0,
#                                  help='plot the prediction for a given bu and dep (yes = 1 | no = 0)')
#         self.initialized = True

#     def parse(self):
#         if not self.initialized:
#             self.initialize()
#         self.opt = self.parser.parse_args()
#         return self.opt



# def map_category_to_vector(all_cat_vector,category):
#     """
#     This functions performs One-Hot Encoding, that is to say it takes a category C as input and 
#     returns a vector filled with 0 except at the spot where the category C is in the vector all_cat_vector.
#     ex : all_cat_vector=[CAT1,CAT2,CAT3,CAT4] and category = CAT 3, the output will be [0,0,1,0].
#     Args:
#         all_cat_vector(np.array or pd.Series): The vector containing all the category that will be used
#         for mapping.
#         category: An element of all_cat_vector to be mapped.
#     """
#     x=np.zeros_like(all_cat_vector)
#     i=np.where(all_cat_vector==category)
#     x[i]=1
#     return x

# def norm(x,max,min):
#     if max==min:
#         return(x-min)
#     else:
#         return(x-min)/(max-min)

# def inversenorm(x,max,min):
#     if max==min:
#         return(x+min)
#     else:
#         return(x*(max-min)+min)

# def correlation(a,y,y_t): 
#     return sklearn.metrics.mean_absolute_error((a*y).reshape(1,-1),y_t.reshape(1,-1))

# def find_bis_year(years):
#     bis_year=[]
#     for year in np.unique(years):
#         cond=False
#         for k in range(0,3):
#             cond=(pd.to_datetime('{}-12-21'.format(year))+pd.DateOffset(weeks=k)).isocalendar().week==53 or cond
#         if cond==True:
#             bis_year.append(year)
#     return bis_year


# class TestDataset():
#     def __init__(self,params):
#         self.samples=[]
#         self.params=params

#     def extract_data(self,past_df,pred_df,bu_feat_df,dic_knn):
#         self.past_df=past_df
#         self.pred_df=pred_df
#         self.bu_feat_df=bu_feat_df
#         self.dic_knn=dic_knn
#         self.past_df["day_id"]=pd.to_datetime(self.past_df["day_id"])
#         self.past_df=self.past_df.join(self.past_df["day_id"].dt.isocalendar())
#         self.pred_df["day_id"]=pd.to_datetime(self.pred_df["day_id"])
#         self.pred_df=self.pred_df.join(self.pred_df["day_id"].dt.isocalendar())
#         bu_num_list=self.pred_df["but_num_business_unit"].unique()
#         for bu in bu_num_list:
#             dep_list=self.pred_df[self.pred_df["but_num_business_unit"]==bu]["dpt_num_department"].unique()
#             bu_info=self.bu_feat_df[self.bu_feat_df["but_num_business_unit"]==bu].iloc[0]
#             region_idr_enc=map_category_to_vector(self.params["region_idr_cat"],bu_info.but_region_idr_region)
#             zod_idr_enc=map_category_to_vector(self.params["zod_idr_cat"],bu_info.zod_idr_zone_dgr)
#             lat=norm(bu_info.but_latitude,self.params["max_lat"],self.params["min_lat"])
#             long=norm(bu_info.but_longitude,self.params["max_long"],self.params["min_long"])
#             for dep in dep_list:
#                 input_knn=np.concatenate([zod_idr_enc,region_idr_enc,np.array([lat,long])]).reshape(1,-1)
#                 output_knn=self.dic_knn[dep].predict(input_knn).reshape(-1)
#                 x_old=np.linspace(0,1,num=128)
#                 x_53=np.linspace(0,1,num=53)
#                 x_52=np.linspace(0,1,num=52)
#                 y_53=np.interp(x_53,x_old,output_knn)
#                 y_52=np.interp(x_52,x_old,output_knn)
#                 self.pred_date=np.min(pred_df["day_id"].unique())
#                 max_turnover=np.max(self.past_df[(self.past_df["but_num_business_unit"]==bu) & (self.past_df["dpt_num_department"]==dep)]["turnover"])
#                 min_turnover=np.min(self.past_df[(self.past_df["but_num_business_unit"]==bu) & (self.past_df["dpt_num_department"]==dep)]["turnover"])
#                 df_x=self.past_df[(self.past_df["but_num_business_unit"]==bu)
#                                     & (self.past_df["dpt_num_department"]==dep)
#                                     & (self.past_df["day_id"]>=self.pred_date-pd.DateOffset(weeks=16))
#                                     & (self.past_df["day_id"]<=self.pred_date-pd.DateOffset(weeks=1))
#                                     ].sort_values(by=["day_id"]).reset_index()
#                 df_y=self.pred_df[(self.pred_df["but_num_business_unit"]==bu)
#                                     & (self.pred_df["dpt_num_department"]==dep)
#                                     & (self.pred_df["day_id"]>=self.pred_date)
#                                     & (self.pred_df["day_id"]<=self.pred_date+pd.DateOffset(weeks=7))
#                                     ].sort_values(by=["day_id"]).reset_index()
#                 no_week_array=np.concatenate([df_x["week"].to_numpy(),df_y["week"].to_numpy()],dim=0)
#                 year_array=np.concatenate([df_x["year"].to_numpy(),df_y["year"].to_numpy()],dim=0)
#                 bis_year=find_bis_year(year_array)
#                 annual_slice=np.zeros(24)
#                 for pos,(w,y) in enumerate(zip(no_week_array,year_array)):
#                     if y in bis_year:
#                         annual_slice[pos]=y_53[w-1]
#                     else:
#                         annual_slice[pos]=y_52[w-1]
#                 raw_x=df_x["turnover"].to_numpy()
#                 max_x=np.max(raw_x)
#                 min_x=np.min(raw_x)
#                 annual_slice_x=annual_slice[0:16]
#                 annual_slice_y=annual_slice[16:24]
#                 normed_raw_x=norm(raw_x,max_x,min_x)
#                 annual_slice_y=norm(annual_slice_y,np.max(annual_slice_x),np.min(annual_slice_x))
#                 annual_slice_x=norm(annual_slice_x,np.max(annual_slice_x),np.min(annual_slice_x))
#                 res = op.minimize_scalar(correlation,args=(annual_slice_x,normed_raw_x))
#                 scale_factor=res.x
#                 annual_slice_x=annual_slice_x*scale_factor
#                 annual_slice_y=annual_slice_y*scale_factor
#                 annual_slice_x=inversenorm(annual_slice_x,max_x,min_x)
#                 annual_slice_y=inversenorm(annual_slice_y,max_x,min_x)
#                 annual_slice_x=norm(annual_slice_x,max_turnover,min_turnover)
#                 annual_slice_y=norm(annual_slice_y,max_turnover,min_turnover)
#                 seq_x=norm(raw_x,max_turnover,min_turnover)
#                 temp_dic={
#                         'normed_x':torch.Tensor(seq_x),
#                         'x':raw_x,
#                         'annual_x':torch.Tensor(annual_slice_x),
#                         'annual_y':torch.Tensor(annual_slice_y),
#                         'normed_pred':None,
#                         'pred':None,
#                         'max_turn':max_turnover,
#                         'min_turn':min_turnover,
#                         'bu_num':bu,
#                         'dep':dep
#                     }
#                 self.samples.append(temp_dic.copy())
#                 i+=1



# class PredSet():
#     def __init__(self):
#         self.past_df=None
#         self.pred_df=None
#         self.bu_feat_df=None
#         self.data_knn=None
#         self.initialized=False
#         self.params={
#                         "max_lat":51.050275,
#                         "min_lat":41.9543,
#                         "max_long":8.7961,
#                         "min_long":-4.4364458576632,
#                         "region_idr_cat":np.array([2,3,4,6,7,8,30,31,32,
#                                                     33,51,52,53,55,64,65,
#                                                     66,69,70,71,72,74,75,
#                                                     107,115,121,134,162,178]),
#                         "zod_idr_cat":np.array([1,3,4,6,10,35,59,72])
#                     }
#     def initialize(self,opt:BaseOptions):
#         self.opt
#         self.past_df=pd.read_csv(os.path.join(opt.dataroot,'/past_data.csv'))
#         self.pred_df=pd.read_csv(os.path.join(opt.dataroot,'/prediction_data.csv'))
#         self.bu_feat_df=pd.read_csv(os.path.join(opt.dataroot,'/bu_feat.csv'))
#         self.data_knn=dataset.Annual_construction_dataset()
#         with open(os.path.join(opt.dataroot,'/data_knn.json'), 'r') as f:
#             self.data_knn.load_from_json(f)
#         self.data_knn.set_data_for_training(False)
#         self.knn_params={
#             73:[9,np.array([2/3,0,1/3])],
#             88:[8,np.array([2/3,0,1/3])],
#             117:[6,np.array([1,0,1])],
#             127:[4,np.array([2/3,0,1])]    
#         }
#         self.dic_knn={
#             73:None,
#             88:None,
#             117:None,
#             127:None
#         }
#         for dep in self.knn_params.keys():
#             X_train=[]
#             Y_train=[]
#             for sample in self.data_knn.samples_per_dep[dep]:
#                 Y_train.append(sample[0])
#                 X_train.append(sample[1])
#             X_train=np.array(X_train)
#             Y_train=np.array(Y_train)
#             self.dic_knn[dep] = KNeighborsRegressor(n_neighbors=self.knn_params[dep][0],
#                                                 weights='distance',
#                                                 n_jobs=-1,
#                                                 metric=lambda a,
#                                                 b:utils.custom_distance(a,b,self.knn_params[dep][1])
#                                                 ).fit(X_train, Y_train)
#         self.dataset_to_nn=TestDataset(self.params)
#         self.dataset_to_nn.extract_data(self.past_df,self.pred_df,self.bu_feat_df,self.dic_knn)
#         self.initialized=True
    
#     def write_prediction(self):
#         self.results_df=self.pred_df.copy()
#         for sample in self.dataset_to_nn.samples:
#             bu=sample["bu_num"]
#             dep=sample["dep"]
#             temp_df=self.results_df[(self.results_df["but_num_business_unit"]==bu) & (self.results_df["dpt_num_department"]==dep)].sort_values(by=["day_id"])
#             days=temp_df["day_id"].to_numpy()
#             for i in range(len(sample["pred"])):
#                 self.results_df.loc[(self.results_df["but_num_business_unit"]==95)
#                                         & (self.results_df["dpt_num_department"]==73)
#                                         & (self.results_df["day_id"]==days[i]),"results"]=sample["pred"][i]
#         self.results_df.to_csv(os.path.join(opt.dataroot,'/results.csv'))
#         print("You will find your results here {}".format(os.path.join(opt.dataroot,'/results.csv')))

# class ModelSet:
#     def __init__(self):
#         self.models_dic={
#                         73:None,
#                         88:None,
#                         117:None,
#                         127:None
#                         }
#         self.models_w_dic={
#                         73:torch.load("./models_weights/weights_73.pth")['model_state_dict'],
#                         88:torch.load("./models_weights/weights_88.pth")['model_state_dict'],
#                         117:torch.load("./models_weights/weights_117.pth")['model_state_dict'],
#                         127:torch.load("./models_weights/weights_127.pth")['model_state_dict']
#                         }
#         self.initialized=False
#     def initialize(self,device):
#         self.device=device
#         deps=[73,88,117,127]
#         for dep in deps:
#             temp_model=lstm.LSTM_Turnover(False,hidden_size=64,num_of_layer=4).to(device)
#             temp_model.load_state_dict(self.models_w_dic[dep])
#             temp_model.eval()
#             self.models_dic[dep]=temp_model
#         self.initialized=True

#     def predict(self,pred_set:PredSet):
#         dataset_to_nn=pred_set.dataset_to_nn
#         for sample in dataset_to_nn.samples:
#             dep=sample["dep"]
#             pred=self.models_dic[dep](sample["normed_x"].float().unsqueeze(0).to(self.device),
#                     sample["annual_x"].float().unsqueeze(0).to(self.device),
#                     sample["annual_y"].float().unsqueeze(0).to(self.device),
#                     8)[0][-8:]
#             pred=pred.detach().cpu().numpy()
#             sample["normed_pred"]=pred
#             sample["pred"]=inversenorm(pred,sample["max_turn"],sample["min_turn"])

from utils import options
from dataset import predset
from model import modelset
import torch

import os

if __name__=='__main__':
    opt = options.BaseOptions().parse()
    if opt.device==1:
        device=torch.device("cuda:0")
        print("Using GPU... \n")
    else:
        device=torch.device("cpu")
        print("Using CPU... \n")
    pred_set=predset.PredSet()
    pred_set.initialize(opt)
    model_set=modelset.ModelSet()
    model_set.initialize(device)
    model_set.predict(pred_set)
    pred_set.write_prediction()