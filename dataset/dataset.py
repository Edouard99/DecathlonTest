import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import tqdm as tqdm

def sequence_is_complete(df:pd.DataFrame,day_id):
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
    end_seq=day_id+pd.DateOffset(weeks=15)
    sequence=df[(df["day_id"]>=day_id) & (df["day_id"]<=end_seq)].sort_values(by=["day_id"]).reset_index()
    valid_seq=[]
    for k in range(0,16):
        valid_seq.append(day_id+pd.DateOffset(weeks=k))
    valid_seq=pd.Series(valid_seq)
    if len(sequence)==16:
        if ((sequence["day_id"]==valid_seq).sum()!=16):
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
    def __init__(self, path_to_folder:str, eval:bool):
        self.samples = []
        self.eval=eval
        self.dataframe=pd.read_csv(path_to_folder+"train.csv")
        self.dataframe_bu=pd.read_csv(path_to_folder+"bu_feat.csv")
        self.dataframe["day_id"]=pd.to_datetime(self.dataframe["day_id"])
        self.dataframe=self.dataframe.join(self.dataframe["day_id"].dt.isocalendar())
        self.discard=[]
        self.max_lat=np.max(self.dataframe_bu["but_latitude"])
        self.min_lat=np.min(self.dataframe_bu["but_latitude"])
        self.max_long=np.max(self.dataframe_bu["but_latitude"])
        self.min_long=np.min(self.dataframe_bu["but_latitude"])
        self.region_idr_cat=np.sort(self.dataframe_bu["but_region_idr_region"].unique())
        self.zod_idr_cat=np.sort(self.dataframe_bu["zod_idr_zone_dgr"].unique())
        bu_num_list=self.dataframe["but_num_business_unit"].unique()
        i=0
        self.dic={}
        for bu in tqdm.tqdm(bu_num_list): # ITERATE ON ALL STORES
            
            dep_list=self.dataframe[(self.dataframe["but_num_business_unit"]==bu)]["dpt_num_department"].unique()
            for dep in dep_list: # ITERATE ON ALL DEPARTMENT OF THE STORE

                temp_df=self.dataframe[(self.dataframe["but_num_business_unit"]==bu) &
                                (self.dataframe["dpt_num_department"]==dep)]
                day_id=np.min(temp_df["day_id"])
                max_day_id=np.max(temp_df["day_id"])

                while day_id<max_day_id: # ITERATE ON ALL DATA AVAILABLE FOR THE DEPARTMENT AND STORE
                    valid,sequence=sequence_is_complete(temp_df,day_id) # Checking that 16 weeks data are available from day_id date
                    if valid: # Data are available
                        df_x=sequence.iloc[0:8]
                        df_y=sequence.iloc[8:16]
                        no_week=(day_id+pd.DateOffset(weeks=8)).isocalendar().week
                        bu_info=self.dataframe_bu[self.dataframe_bu["but_num_business_unit"]==bu].iloc[0]
                        region_idr=bu_info.but_region_idr_region
                        zod_idr=bu_info.zod_idr_zone_dgr
                        region_idr_enc=map_category_to_vector(self.region_idr_cat,region_idr).tolist()
                        zod_idr_enc=map_category_to_vector(self.zod_idr_cat,zod_idr).tolist()
                        lat=bu_info.but_latitude
                        lat=(lat-self.min_lat)/(self.max_lat-self.min_lat)
                        long=bu_info.but_longitude
                        long=(long-self.min_long)/(self.max_long-self.min_long)
                        sequence_x=df_x["turnover"].to_numpy().tolist()
                        sequence_y=df_y["turnover"].to_numpy().tolist()
                        temp_dic={
                            'seq_x':sequence_x,
                            'seq_y':sequence_y,
                            'bu_num':bu,
                            'dep':dep,
                            'day_id':day_id,
                            'no_week':no_week,
                            'region_idr_enc':region_idr_enc,
                            'zod_idr_enc':zod_idr_enc,
                            'lat':lat,
                            'long':long
                        }
                        self.samples.append(temp_dic.copy() )
                        self.dic[i]=temp_dic.copy()
                        i+=1
                    else: # Data are not available
                        self.discard.append((bu,dep,day_id))
                    day_id=day_id+pd.DateOffset(weeks=1) # Go to the next data available
    
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        return self.samples[id]