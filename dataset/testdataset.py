import pandas as pd
import scipy.optimize as op
from utils import utils
import numpy as np
import torch
import tqdm
class TestDataset():
    def __init__(self,params):
        self.samples=[]
        self.params=params
        

    def extract_data(self,past_df,pred_df,bu_feat_df,dic_knn):
        self.padded=0
        self.past_df=past_df
        self.pred_df=pred_df
        self.bu_feat_df=bu_feat_df
        self.dic_knn=dic_knn
        self.past_df["day_id"]=pd.to_datetime(self.past_df["day_id"])
        self.past_df=self.past_df.join(self.past_df["day_id"].dt.isocalendar())
        self.pred_df["day_id"]=pd.to_datetime(self.pred_df["day_id"])
        self.pred_df=self.pred_df.join(self.pred_df["day_id"].dt.isocalendar())
        bu_num_list=self.pred_df["but_num_business_unit"].unique()
        print("Extracting data... \n")
        for bu in tqdm.tqdm(bu_num_list):
            dep_list=self.pred_df[self.pred_df["but_num_business_unit"]==bu]["dpt_num_department"].unique()
            bu_info=self.bu_feat_df[self.bu_feat_df["but_num_business_unit"]==bu].iloc[0]
            region_idr_enc=utils.map_category_to_vector(self.params["region_idr_cat"],bu_info.but_region_idr_region)
            zod_idr_enc=utils.map_category_to_vector(self.params["zod_idr_cat"],bu_info.zod_idr_zone_dgr)
            lat=utils.norm(bu_info.but_latitude,self.params["max_lat"],self.params["min_lat"])
            long=utils.norm(bu_info.but_longitude,self.params["max_long"],self.params["min_long"])
            for dep in dep_list:
                input_knn=np.concatenate([zod_idr_enc,region_idr_enc,np.array([lat,long])]).reshape(1,-1)
                output_knn=self.dic_knn[dep].predict(input_knn).reshape(-1)
                x_old=np.linspace(0,1,num=128)
                x_53=np.linspace(0,1,num=53)
                x_52=np.linspace(0,1,num=52)
                y_53=np.interp(x_53,x_old,output_knn)
                y_52=np.interp(x_52,x_old,output_knn)
                self.pred_date=np.min(pred_df["day_id"].unique())
                max_turnover=np.max(self.past_df[(self.past_df["but_num_business_unit"]==bu) & (self.past_df["dpt_num_department"]==dep)]["turnover"])
                min_turnover=np.min(self.past_df[(self.past_df["but_num_business_unit"]==bu) & (self.past_df["dpt_num_department"]==dep)]["turnover"])
                df_x=self.past_df[(self.past_df["but_num_business_unit"]==bu)
                                    & (self.past_df["dpt_num_department"]==dep)
                                    & (self.past_df["day_id"]>=self.pred_date-pd.DateOffset(weeks=16))
                                    & (self.past_df["day_id"]<=self.pred_date-pd.DateOffset(weeks=1))
                                    ].sort_values(by=["day_id"]).reset_index()
                if len(df_x)!=16:
                    self.padded+=1
                    temp_min_date=np.min(df_x["day_id"])
                    for k in range(16-len(df_x)):
                        temp_date=temp_min_date-pd.DateOffset(weeks=k+1)
                        new_line = pd.DataFrame([{'index':pd.NaT,
                                        'day_id':temp_date,
                                        'but_num_business_unit':bu,
                                        'dpt_num_department':dep,
                                        'turnover':0,
                                        'year':temp_date.isocalendar().year,
                                        'week':temp_date.isocalendar().week,
                                        'day':temp_date.isocalendar().weekday}])
                        df_x = pd.concat([df_x, new_line], axis=0, ignore_index=True)
                df_x=df_x.sort_values(by=["day_id"]).reset_index()
                df_y=self.pred_df[(self.pred_df["but_num_business_unit"]==bu)
                                    & (self.pred_df["dpt_num_department"]==dep)
                                    & (self.pred_df["day_id"]>=self.pred_date)
                                    & (self.pred_df["day_id"]<=self.pred_date+pd.DateOffset(weeks=7))
                                    ].sort_values(by=["day_id"]).reset_index()
                no_week_array=np.concatenate([df_x["week"].to_numpy(),df_y["week"].to_numpy()])
                year_array=np.concatenate([df_x["year"].to_numpy(),df_y["year"].to_numpy()])
                bis_year=utils.find_bis_year(year_array)
                annual_slice=np.zeros(24)
                for pos,(w,y) in enumerate(zip(no_week_array,year_array)):
                    if y in bis_year:
                        annual_slice[pos]=y_53[w-1]
                    else:
                        annual_slice[pos]=y_52[w-1]
                raw_x=df_x["turnover"].to_numpy()
                max_x=np.max(raw_x)
                min_x=np.min(raw_x)
                annual_slice_x=annual_slice[0:16]
                annual_slice_y=annual_slice[16:24]
                normed_raw_x=utils.norm(raw_x,max_x,min_x)
                annual_slice_y=utils.norm(annual_slice_y,np.max(annual_slice_x),np.min(annual_slice_x))
                annual_slice_x=utils.norm(annual_slice_x,np.max(annual_slice_x),np.min(annual_slice_x))
                res = op.minimize_scalar(utils.correlation,args=(annual_slice_x,normed_raw_x))
                scale_factor=res.x
                annual_slice_x=annual_slice_x*scale_factor
                annual_slice_y=annual_slice_y*scale_factor
                annual_slice_x=utils.inversenorm(annual_slice_x,max_x,min_x)
                annual_slice_y=utils.inversenorm(annual_slice_y,max_x,min_x)
                annual_slice_x=utils.norm(annual_slice_x,max_turnover,min_turnover)
                annual_slice_y=utils.norm(annual_slice_y,max_turnover,min_turnover)
                seq_x=utils.norm(raw_x,max_turnover,min_turnover)
                temp_dic={
                        'normed_x':torch.Tensor(seq_x),
                        'x':raw_x,
                        'annual_x':torch.Tensor(annual_slice_x),
                        'annual_y':torch.Tensor(annual_slice_y),
                        'normed_pred':None,
                        'pred':None,
                        'max_turn':max_turnover,
                        'min_turn':min_turnover,
                        'bu_num':bu,
                        'dep':dep
                    }
                self.samples.append(temp_dic.copy())
        print("Extraction done, had to pad {} data. \n".format(self.padded))