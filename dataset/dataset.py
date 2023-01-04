import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import tqdm as tqdm
import json
import torch
from numpy.lib.stride_tricks import sliding_window_view

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,type(pd.to_datetime('2022-01-01'))):
            return str(obj)
        return super(NpEncoder, self).default(obj)


def inverse_diff(first,diff):
    return np.concatenate(([first], diff)).cumsum()

def sequence_is_complete(df:pd.DataFrame,day_id,number_of_weeks):
    """
    This function check if a given dataframe contains the data for a date D and
    for the next 15 weeks to obtain a sequence of data from the dates : (D,D+1week,D+2weeks,...,D+15weeks).
    A complete sequence is called a "valid sequence".
    This functions returns a boolean assessing the validity of the sequence and the sequence.
    Args:
        df(pd.DataFrame): A dataframe (in our case restrained to 1 business unit and 1 department).
        day_id(Timestamp): A timestamp that will correspond to the date D.
    """
    
    valid=True
    end_seq=day_id+pd.DateOffset(weeks=number_of_weeks)
    sequence=df[(df["day_id"]>=day_id) & (df["day_id"]<=end_seq)].sort_values(by=["day_id"]).reset_index()
    valid_seq=[]
    for k in range(0,number_of_weeks+1):
        valid_seq.append(day_id+pd.DateOffset(weeks=k))
    valid_seq=pd.Series(valid_seq)
    if len(sequence)==number_of_weeks+1:
        if ((sequence["day_id"]==valid_seq).sum()!=number_of_weeks+1):
            valid=False
    else:
        valid=False
    return valid,sequence

def map_category_to_vector(all_cat_vector,category):
    """
    This functions performs One-Hot Encoding, that is to say it takes a category C as input and 
    returns a vector filled with 0 except at the spot where the category C is in the vector all_cat_vector.
    ex : all_cat_vector=[CAT1,CAT2,CAT3,CAT4] and category = CAT 3, the output will be [0,0,1,0].
    Args:
        all_cat_vector(np.array or pd.Series): The vector containing all the category that will be used
        for mapping.
        category: An element of all_cat_vector to be mapped.
    """
    x=np.zeros_like(all_cat_vector)
    i=np.where(all_cat_vector==category)
    x[i]=1
    return x

def norm(x,max,min):
    if max==min:
        return(x-min)
    else:
        return(x-min)/(max-min)

def inversenorm(x,max,min):
    if max==min:
        return(x+min)
    else:
        return(x*(max-min)+min)

class Annual_construction_dataset(Dataset):
    def __init__(self):
        self.samples=[]
        self.dep=None
        self.samples_for_nn=[]
        self.samples_dep_73=[]
        self.samples_dep_88=[]
        self.samples_dep_117=[]
        self.samples_dep_127=[]
        self.discard=0

    def load_from_df(self, dataframe_turnover, data_frame_bu):
        self.samples=[]
        self.discard=0
        self.dataframe=dataframe_turnover
        self.dataframe_bu=data_frame_bu
        self.dataframe["day_id"]=pd.to_datetime(self.dataframe["day_id"])
        self.dataframe=self.dataframe.join(self.dataframe["day_id"].dt.isocalendar())
        
        self.max_lat=np.max(self.dataframe_bu["but_latitude"])
        self.min_lat=np.min(self.dataframe_bu["but_latitude"])
        self.max_long=np.max(self.dataframe_bu["but_longitude"])
        self.min_long=np.min(self.dataframe_bu["but_longitude"])
        self.region_idr_cat=np.sort(self.dataframe_bu["but_region_idr_region"].unique())
        self.zod_idr_cat=np.sort(self.dataframe_bu["zod_idr_zone_dgr"].unique())
        self.dep_cat=np.sort(self.dataframe["dpt_num_department"].unique())
        bu_num_list=self.dataframe["but_num_business_unit"].unique()
        self.dic={}
        i=0
        for bu in tqdm.tqdm(bu_num_list): # ITERATE ON ALL STORES
            dep_list=self.dataframe[(self.dataframe["but_num_business_unit"]==bu)]["dpt_num_department"].unique()
            bu_info=self.dataframe_bu[self.dataframe_bu["but_num_business_unit"]==bu].iloc[0]
            region_idr=bu_info.but_region_idr_region
            zod_idr=bu_info.zod_idr_zone_dgr
            region_idr_enc=map_category_to_vector(self.region_idr_cat,region_idr).tolist()
            zod_idr_enc=map_category_to_vector(self.zod_idr_cat,zod_idr).tolist()
            lat=bu_info.but_latitude
            lat=(lat-self.min_lat)/(self.max_lat-self.min_lat)
            long=bu_info.but_longitude
            long=(long-self.min_long)/(self.max_long-self.min_long)
            for dep in dep_list: # ITERATE ON ALL DEPARTMENT OF THE STORE
                temp=self.dataframe[(self.dataframe["but_num_business_unit"]==bu) & (self.dataframe["dpt_num_department"]==dep)]
                years=temp["year"].unique()
                L=[]
                for year in years:
                    if year==2015:
                        final_week=53
                    else:
                        final_week=52
                    temp_year=temp[temp["year"]==year].sort_values(by="day_id")
                    first_week_of_year=temp_year.head(1)["week"].item()
                    last_week_of_year=temp_year.tail(1)["week"].item()
                    if ((first_week_of_year==1) & (last_week_of_year==final_week)):
                        turnover_data=temp_year["turnover"].to_numpy()
                        turnover_data_norm=norm(turnover_data,np.max(turnover_data),np.min(turnover_data))
                        y_old=np.array(turnover_data_norm)
                        x_old=np.linspace(1,len(y_old),num=len(y_old))
                        x=np.linspace(1,len(y_old),num=128)
                        y=np.interp(x,x_old,y_old)
                        L.append(y)
                if len(L)==0:
                    self.discard+=1
                else:
                    y_mean=np.mean(L,axis=0)
                    temp_dic={
                        "bu":bu,
                        "years":np.array(years),
                        "dep":dep,
                        "raw_data":L,
                        "data":y_mean,
                        'region_idr':region_idr,
                        'zod_idr':zod_idr,
                        'region_idr_enc':region_idr_enc,
                        'zod_idr_enc':zod_idr_enc,
                        'raw_lat':bu_info.but_latitude,
                        'raw_long':bu_info.but_longitude,
                        'lat':lat,
                        'long':long
                    }
                    self.samples.append(temp_dic.copy())
                    self.dic[i]=temp_dic.copy()
                    i+=1
    
    def save_ds_to_json(self,file):
        json.dump(self.dic, file,cls=NpEncoder)

    def load_from_json(self,file):
        self.dic=json.load(file)
        for key in self.dic.keys():
            self.samples.append(self.dic[key])

    def set_data_for_training(self,free_sample_memory):
        self.samples_dep_117=[]
        self.samples_dep_73=[]
        self.samples_dep_127=[]
        self.samples_dep_88=[]
        self.samples_for_nn=[]
        for id_s,sample in enumerate(self.samples):
            y=np.array(sample["data"])
            dep=sample["dep"]
            x=np.concatenate([np.array(sample["zod_idr_enc"]),np.array(sample["region_idr_enc"]),np.array([sample["lat"],sample["long"]])])
            temp_dic={
                                'y':y,
                                'x':x,
                                'id':id_s
                            }
            self.samples_for_nn.append(temp_dic.copy())
            if dep==73:
                self.samples_dep_73.append([y,x,id_s])
            if dep==88:
                self.samples_dep_88.append([y,x,id_s])
            if dep==117:
                self.samples_dep_117.append([y,x,id_s])
            if dep==127:
                self.samples_dep_127.append([y,x,id_s])
        if free_sample_memory:
            self.samples=[]


    def __len__(self):
        if self.dep==73:
            return len(self.samples_dep_73)
        elif self.dep==88:
            return len(self.samples_dep_88)
        elif self.dep==117:
            return len(self.samples_dep_117)
        elif self.dep==127:
            return len(self.samples_dep_127)
        else:
            return len(self.samples_for_nn)


    def __getitem__(self, id):
        if self.dep==73:
            return self.samples_dep_73[id]
        elif self.dep==88:
            return self.samples_dep_88[id]
        elif self.dep==117:
            return self.samples_dep_117[id]
        elif self.dep==127:
            return self.samples_dep_127[id]
        else:
            return self.samples_for_nn[id]




class Turnover_dataset(Dataset):
    """
    Generate a dataset of turnover and additional data.
    Each element of the data consist in a dictionnary containing:
        - 'seq_x' : list of turnovers for a given business unit between date "day_id" and "day_id+7weeks".
            The turnovers are given per week so the list will be a 8 element list.
            The turnovers are sorted by date (index 0 = turnover on week of "day_id";
            index 7 = turnover on "day_id+7weeks")
        - 'seq_y' : list of turnovers for a given business unit between date "day_id+8weeks" and "day_id+15weeks".
            The turnovers are given per week so the list will be a 8 element list.
            The turnovers are sorted by date (index 0 = turnover on week of "day_id+8weeks";
            index 7 = turnover on "day_id+15weeks")
        - 'bu_num' : business unit number from where the data of an element where collected
        - 'dep' : department number from where the data of an element where collected
        - 'day_id' : day_id from where the data are collected (16weeks)
        - 'no_week' : number of the week of the year corresponding to the day_id week
        - 'region_idr_enc' : One-Hot encoded vector corresponding to but_region_idr_region of the business unit
        - 'zod_idr_enc': One-Hot encoded vector corresponding to zod_idr_zone_dgr of the business unit
        - 'lat' : normalized latitude of the business unit (normalized using min and max of latitude over
            all the Decathlon stores)
        - 'long' : normalized longitude of the business unit (normalized using min and max of longitude over
            all the Decathlon stores)

    Args:
        path_to_folder(str): path to the folder that contains both train.csv and bu_feat.csv
        eval(bool): boolean used to use the dataset in training/eval mode.
    """
    def __init__(self):
        self.samples=[]
        self.samples_for_nn= []
        self.eval=False
        self.dataframe=None
        self.dataframe_bu=None
        self.discard=[]
        self.max_lat=None
        self.min_lat=None
        self.max_long=None
        self.min_long=None
        self.region_idr_cat=None
        self.zod_idr_cat=None
        self.dic={}
        self.samples_dep_73=[]
        self.samples_dep_88=[]
        self.samples_dep_117=[]
        self.samples_dep_127=[]
    
    def load_from_df(self, dataframe_turnover, data_frame_bu):
        
        self.eval=False
        self.dep=None
        self.dataframe=dataframe_turnover
        self.dataframe_bu=data_frame_bu
        self.dataframe["day_id"]=pd.to_datetime(self.dataframe["day_id"])
        self.dataframe=self.dataframe.join(self.dataframe["day_id"].dt.isocalendar())
        self.discard=[]
        self.max_lat=np.max(self.dataframe_bu["but_latitude"])
        self.min_lat=np.min(self.dataframe_bu["but_latitude"])
        self.max_long=np.max(self.dataframe_bu["but_longitude"])
        self.min_long=np.min(self.dataframe_bu["but_longitude"])
        self.region_idr_cat=np.sort(self.dataframe_bu["but_region_idr_region"].unique())
        self.zod_idr_cat=np.sort(self.dataframe_bu["zod_idr_zone_dgr"].unique())
        self.dep_cat=np.sort(self.dataframe["dpt_num_department"].unique())
        bu_num_list=self.dataframe["but_num_business_unit"].unique()
        i=0
        self.dic={}
        for bu in tqdm.tqdm(bu_num_list): # ITERATE ON ALL STORES
            dep_list=self.dataframe[(self.dataframe["but_num_business_unit"]==bu)]["dpt_num_department"].unique()
            bu_info=self.dataframe_bu[self.dataframe_bu["but_num_business_unit"]==bu].iloc[0]
            region_idr=bu_info.but_region_idr_region
            zod_idr=bu_info.zod_idr_zone_dgr
            region_idr_enc=map_category_to_vector(self.region_idr_cat,region_idr).tolist()
            zod_idr_enc=map_category_to_vector(self.zod_idr_cat,zod_idr).tolist()
            lat=bu_info.but_latitude
            lat=(lat-self.min_lat)/(self.max_lat-self.min_lat)
            long=bu_info.but_longitude
            long=(long-self.min_long)/(self.max_long-self.min_long)
            for dep in dep_list: # ITERATE ON ALL DEPARTMENT OF THE STORE

                temp_df=self.dataframe[(self.dataframe["but_num_business_unit"]==bu) &
                                (self.dataframe["dpt_num_department"]==dep)]
                day_id=np.min(temp_df["day_id"])
                max_day_id=np.max(temp_df["day_id"])
                dep_enc=map_category_to_vector(self.dep_cat,dep).tolist()
                while day_id<max_day_id: # ITERATE ON ALL DATA AVAILABLE FOR THE DEPARTMENT AND STORE
                    valid,sequence=sequence_is_complete(temp_df,day_id,23) # Checking that 16 weeks data are available from day_id date
                    if valid: # Data are available
                        no_week=(day_id).isocalendar().week
                        year=(day_id).isocalendar().year
                        if year==2015:
                            no_week=no_week/53
                        else:
                            no_week=no_week/52
                        raw_xy=sequence.iloc[0:24]["turnover"].to_numpy()
                        raw_x=raw_xy[0:16]
                        raw_y=raw_xy[16:24]
                        diff_xy=np.diff(raw_xy)
                        diff_x=diff_xy[0:15]
                        diff_y=diff_xy[15:23]
                        max_x=np.max(diff_x)
                        min_x=np.min(diff_x)
                        seq_x=norm(diff_x,max_x,min_x)
                        seq_y=norm(diff_y,max_x,min_x)
                        seq_x_slide=sliding_window_view(seq_x,5)
                        
                        temp_dic={
                            'seq_x':seq_x,
                            'raw_x':raw_x,
                            'seq_y':seq_y,
                            'raw_y':raw_y,
                            'raw_xy':raw_xy,
                            'seq_x_full':seq_x,
                            'seq_y_full':seq_y,
                            'seq_x_slide':seq_x_slide,
                            'seq_y_one':seq_y[0],
                            'first_turn':raw_xy[0],
                            'max_x':max_x,
                            'min_x':min_x,
                            'bu_num':bu,
                            'dep':dep,
                            'dep_enc':dep_enc,
                            'day_id':day_id,
                            'no_week':no_week,
                            'region_idr':region_idr,
                            'zod_idr':zod_idr,
                            'region_idr_enc':region_idr_enc,
                            'zod_idr_enc':zod_idr_enc,
                            'raw_lat':bu_info.but_latitude,
                            'raw_long':bu_info.but_longitude,
                            'lat':lat,
                            'long':long
                        }
                        self.samples.append(temp_dic.copy())
                        self.dic[i]=temp_dic.copy()
                        i+=1
                    else: # Data are not available
                        self.discard.append((bu,dep,day_id))
                    day_id=day_id+pd.DateOffset(weeks=1) # Go to the next data available 

    def save_ds_to_json(self,file):
        json.dump(self.dic, file,cls=NpEncoder)

    def load_from_json(self,file):
        self.dic=json.load(file)
        for key in self.dic.keys():
            self.samples.append(self.dic[key])

    def set_data_for_training(self,free_sample_memory):
        self.samples_dep_117=[]
        self.samples_dep_73=[]
        self.samples_dep_127=[]
        self.samples_dep_88=[]
        self.samples_for_nn=[]
        for id_s,sample in enumerate(self.samples):
            x=torch.tensor(sample["seq_x"])
            x_slide=torch.tensor(sample["seq_x_slide"])
            y=torch.tensor(sample["seq_y"])
            y_one=torch.tensor(sample["seq_y_one"])
            zod=torch.tensor(sample["zod_idr_enc"])
            idr=torch.tensor(sample["region_idr_enc"])
            dep=torch.tensor(sample["dep"])
            dep_enc=torch.tensor(sample["dep_enc"])
            add_input=torch.tensor(np.array([sample["lat"],sample["long"],sample["no_week"]]))
            # mean_x=torch.tensor(sample["mean_x"])
            # std_x=torch.tensor(sample["std_x"])
            temp_dic={
                                'x':x,
                                'y':y,
                                'x_slide':x_slide,
                                'y_one':y_one,
                                'zod':zod,
                                'idr':idr,
                                'dep':dep,
                                'dep_enc':dep_enc,
                                'add_input':add_input,
                                'id':id_s
                            }
            self.samples_for_nn.append(temp_dic.copy())
            if dep==73:
                self.samples_dep_73.append(temp_dic.copy())
            if dep==88:
                self.samples_dep_88.append(temp_dic.copy())
            if dep==117:
                self.samples_dep_117.append(temp_dic.copy())
            if dep==127:
                self.samples_dep_127.append(temp_dic.copy())
        if free_sample_memory:
            self.samples=[]

    def __len__(self):
        if self.dep==73:
            return len(self.samples_dep_73)
        elif self.dep==88:
            return len(self.samples_dep_88)
        elif self.dep==117:
            return len(self.samples_dep_117)
        elif self.dep==127:
            return len(self.samples_dep_127)
        else:
            return len(self.samples_for_nn)


    def __getitem__(self, id):
        if self.dep==73:
            return self.samples_dep_73[id]
        elif self.dep==88:
            return self.samples_dep_88[id]
        elif self.dep==117:
            return self.samples_dep_117[id]
        elif self.dep==127:
            return self.samples_dep_127[id]
        else:
            return self.samples_for_nn[id]