import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import tqdm as tqdm
import json
import torch
from numpy.lib.stride_tricks import sliding_window_view
import scipy.optimize as op
from utils import utils

class NpEncoder(json.JSONEncoder):
    """
    Json Encoder used to write dataset object to a json file

    """
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

def sequence_is_complete(df:pd.DataFrame,day_id,number_of_weeks):
    """
    This function check if a given dataframe contains the data for a date D and
    for the next N weeks to obtain a sequence of data from the dates : (D,D+1week,D+2weeks,...,D+Nweeks).
    A complete sequence is called a "valid sequence".
    This functions returns a boolean assessing the validity of the sequence and the sequence.
    Args:
        df(pd.DataFrame): A dataframe (in our case restrained to 1 business unit and 1 department).
        day_id(Timestamp): A timestamp that will correspond to the date D.
        number_of_weeks(int): N value.
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

# def inverse_diff(first,diff):
#     return np.concatenate(([first], diff)).cumsum()

# def correlation(a,y,y_t): 
#     return sklearn.metrics.mean_absolute_error((a*y).reshape(1,-1),y_t.reshape(1,-1))

class Annual_construction_dataset(Dataset):
    """
    Dataset object that contains average annual turnover timeserie for each business unit and each department.
    It is based on dataframe_turnover data.

    With this dataset a knn model can be trained to predict average annual turnover timeserie given departement and 
    business unit features.

    The dataset is based on torch Dataset object so it can be used with a torch dataloader, it will returns data preprocessed for a knn,
    depending on dep attribute the dataloader will return data related to a given department (or not).

    Custom Methods:
        load_from_df : loads the dataset from dataframe_turnover and from dataframe_bu.
        save_ds_to_json : saves the loaded dataset to a json file.
        load_from_json : loads dataset from json file.
        set_data_for_training : converts the loaded dataset to a format used for training and prediction of a knn model.


    Attributes:
        samples : list of data, each element is a dictionnary based on this format :
                    dic={
                        "bu": business unit id,
                        "years": array of years used to compute the average annual average turnover timeserie,
                        "dep": department id,
                        "raw_data": list of annual turnover timeserie (each element correspond to a year),
                        "data": average annual average turnover timeserie for this business unit and department,
                        'region_idr': but_region_idr_region of the business unit,
                        'zod_idr': zod_idr_zone_dgr of the business unit,
                        'region_idr_enc': but_region_idr_region one-hot encoded with function utils.map_category_to_vector(),
                        'zod_idr_enc': zod_idr_zone_dgr one-hot encoded with function utils.map_category_to_vector(),
                        'raw_lat':but_latitude of the business unit,
                        'raw_long':but_longitude of the business unit,
                        'lat': normalized latitude of the business unit (using maximum and minimum latitude of all business units),
                        'long': normalized longitude of the business unit (using maximum and minimum longitude of all business units)
                    }
        dic : dic of data, each element associated with the key i correspond to samples[i], it is used for json saving and loading.
        samples_for_knn: list of data preprocessed for knn, element k is processed from samples[k]
                    and results in a dictionnary based on this format :
                    preprocess_dic={
                            'y': array of average annual average turnover timeserie for this business unit and department,
                            'x': array of the concatenation of (zod_idr_enc,region_idr_enc,lat,long)
                            'id': k
                        }
        samples_per_dep: dictionnary where each element associated to the id of a departement is a list of preprocessed dic.
                    for dep=D samples_per_dep[D] returns the list of all preprocessed_dic related to department D constructed
                    from samples list.
        dep: dataset department mode, if dep is set to an id of department D that correspond to a key of samples_per_dep
                    then the dataset will only return preprocessed_dic related to department D.
                    ex : If you want to have only samples related to department 73, set dep to 73.
                         If you want to have all samples not regarding the department then set dep to None.
        discard: number of pair (business unit id, department id) for which it was impossible to compute average annual turnover
                    due to the lack of data.
        dataframe: dataframe of annual turnover used to compute samples.
        dataframe_bu: dataframe of business unit data used to compute samples.
        max_lat: maximum latitude of all business units in dataframe_bu.
        min_lat: minimum latitude of all business units in dataframe_bu.
        max_long: maximum longitude of all business units in dataframe_bu.
        min_long: minimum longitude of all business units in dataframe_bu.
        region_idr_cat: array of all unique but_region_idr_region in dataframe_bu used to one-hot encoding.
        zod_idr_cat: array of all unique zod_idr_zone_dgr in dataframe_bu used to one-hot encoding.
        dep_cat: array of all unique department id in dataframe_bu.
    """
    def __init__(self):
        self.samples=[]
        self.dep=None
        self.samples_for_knn=[]
        self.samples_per_dep={73:[],88:[],117:[],127:[]}
        self.discard=0

    def load_from_df(self, dataframe_turnover, dataframe_bu):
        """
        Load dataset from 2 given dataframes into samples and dic. Each element is a dictionnary on this format:
            dic={
                        "bu": business unit id,
                        "years": array of years used to compute the average annual average turnover timeserie,
                        "dep": department id,
                        "raw_data": list of annual turnover timeserie (each element correspond to a year),
                        "data": average annual average turnover timeserie for this business unit and department,
                        'region_idr': but_region_idr_region of the business unit,
                        'zod_idr': zod_idr_zone_dgr of the business unit,
                        'region_idr_enc': but_region_idr_region one-hot encoded with function utils.map_category_to_vector(),
                        'zod_idr_enc': zod_idr_zone_dgr one-hot encoded with function utils.map_category_to_vector(),
                        'raw_lat':but_latitude of the business unit,
                        'raw_long':but_longitude of the business unit,
                        'lat': normalized latitude of the business unit (using maximum and minimum latitude of all business units),
                        'long': normalized longitude of the business unit (using maximum and minimum longitude of all business units)
                    }
            Note that the "data" element is a 128 sized array that respresent average annual turnover over all the years in
            "years" element. Index 0 of this array corresponds to start of the year, index -1 correponds to the end of the year.
        Args:
            - dataframe_turnover(pd.Dataframe) :  a dataframe containing columns : ["day_id],["but_num_business_unit"],
                ["dpt_num_department"],["turnover"]. Its data will be used for computation of annual average turnover.
            - dataframe_bu(pd.Dataframe) :  a dataframe containing columns : ["but_num_business_unit"],["but_latitude"],
                ["but_longitude"],["but_region_idr_region"],["zod_idr_zone_dgr"]. 
                Its data will be used for computation of annual average turnover.         
        """

        self.samples=[]
        self.dic={}
        self.discard=0
        self.dataframe=dataframe_turnover
        self.dataframe_bu=dataframe_bu
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
        i=0
        for bu in tqdm.tqdm(bu_num_list): # ITERATE ON ALL STORES
            dep_list=self.dataframe[(self.dataframe["but_num_business_unit"]==bu)]["dpt_num_department"].unique()
            bu_info=self.dataframe_bu[self.dataframe_bu["but_num_business_unit"]==bu].iloc[0]
            region_idr=bu_info.but_region_idr_region
            zod_idr=bu_info.zod_idr_zone_dgr
            region_idr_enc=utils.map_category_to_vector(self.region_idr_cat,region_idr).tolist() #One-hot encoding
            zod_idr_enc=utils.map_category_to_vector(self.zod_idr_cat,zod_idr).tolist() #One-hot encoding
            lat=bu_info.but_latitude
            lat=utils.norm(lat,self.max_lat,self.min_lat) #Normalizing
            long=bu_info.but_longitude
            long=utils.norm(long,self.max_long,self.min_long) #Normalizing
            for dep in dep_list: # ITERATE ON ALL DEPARTMENT OF THE STORE
                temp=self.dataframe[(self.dataframe["but_num_business_unit"]==bu) & (self.dataframe["dpt_num_department"]==dep)]
                years=temp["year"].unique()
                L=[]
                for year in years:
                    if year==2015: #Handle year with 53 weeks
                        final_week=53
                    else:
                        final_week=52
                    temp_year=temp[temp["year"]==year].sort_values(by="day_id")
                    first_week_of_year=temp_year.head(1)["week"].item()
                    last_week_of_year=temp_year.tail(1)["week"].item()
                    if ((first_week_of_year==1) & (last_week_of_year==final_week)): #Check if the turnover timeserie correspond to a complete year
                        turnover_data=temp_year["turnover"].to_numpy()
                        turnover_data_norm=utils.norm(turnover_data,np.max(turnover_data),np.min(turnover_data)) #Normalizing
                        y_old=np.array(turnover_data_norm)
                        x_old=np.linspace(1,len(y_old),num=len(y_old))
                        x=np.linspace(1,len(y_old),num=128) 
                        y=np.interp(x,x_old,y_old) # Interpolate annual turnover timeserie into 128 points
                        L.append(y)
                if len(L)==0:
                    self.discard+=1
                else:
                    y_mean=np.mean(L,axis=0) # Compute average annual turnover timeserie
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
        """
        Saves dataset dic attribute into a json file using a custom NpEncoder to handle numpy.
        Args:
            - file: an open json file
        """
        json.dump(self.dic, file,cls=NpEncoder)

    def load_from_json(self,file):
        """
        Loads dataset from a json file into dic and samples attributes.
        Args:
            - file: an open json file
        """
        self.dic=json.load(file)
        for key in self.dic.keys():
            self.samples.append(self.dic[key])

    def set_data_for_training(self,free_sample_memory):
        """
        Preprocess the data from samples and stores it into samples_for_knn and samples_per_dep attributes.
        Args:
            - free_sample_memory(boolean) : if True the samples attribute will be set to an empty list.
        
        """
        self.samples_per_dep={73:[],88:[],117:[],127:[]}
        self.samples_for_knn=[]
        for id_s,sample in enumerate(self.samples):
            y=np.array(sample["data"])
            dep=sample["dep"]
            x=np.concatenate([np.array(sample["zod_idr_enc"]),np.array(sample["region_idr_enc"]),np.array([sample["lat"],sample["long"]])])
            temp_dic={
                                'y':y, # average annual turnover timeserie
                                'x':x, # X is the feature vector, the input of our knn model
                                'id':id_s
                            }
            self.samples_for_knn.append(temp_dic.copy())
            self.samples_per_dep[dep].append([y,x,id_s])
        if free_sample_memory:
            self.samples=[]


    def __len__(self):
        if self.dep in [73,88,117,127]:
            return len(self.samples_per_dep[self.dep])
        else:
            return len(self.samples_for_knn)


    def __getitem__(self, id):
        if self.dep in [73,88,117,127]:
            return self.samples_per_dep[self.dep][id]
        else:
            return self.samples_for_knn[id]




class Turnover_dataset(Dataset):
    """
    Dataset object that contains turnover data to feed and train the neural network to make a prediction.
    It is based on dataframe_turnover data.

    With this dataset a neural network can be trained to predict from 1 to 8 next weeks turnovers given the last 16 weeks
    turnovers and the average annual turnover corresponding to the business unit & department.

    The dataset is based on torch Dataset object so it can be used with a torch dataloader, it will returns data preprocessed for a knn,
    depending on dep attribute the dataloader will return data related to a given department (or not).

    Custom Methods:
        load_from_df : loads the dataset from dataframe_turnover and from dataframe_bu.
        save_ds_to_json : saves the loaded dataset to a json file.
        load_from_json : loads dataset from json file.
        set_data_for_training : converts the loaded dataset to a format used for training and prediction of a knn model.


    Attributes:
        samples : list of data, each element is a dictionnary based on this format :
                    dic={
                        'seq_x': timeserie of the last 16 weeks turnovers (of a given business unit and department) 
                            normalized by the maximum and minimum turnovers of this business unit and department ever made,
                        'raw_x': timeserie of the last 16 weeks turnovers (of a given business unit and department),
                        'seq_y':timeserie of the next 8 weeks turnovers (of a given business unit and department) 
                            normalized by the maximum and minimum turnovers of this business unit and department ever made,
                        'raw_y':timeserie of the next 8 weeks turnovers (of a given business unit and department),
                        'raw_xy':timeserie of the last 16 weeks and the next 8 weeks turnovers (of a given business unit and department),
                        'annual_slice_x':timeserie of the last 16 weeks average annual turnovers (of a given business unit and department)
                            scaled to seq_x data,
                        'annual_slice_y':timeserie of the next 8 weeks average annual turnovers (of a given business unit and department)
                            scaled to seq_x data,
                        'scale_factor':scale factor used for scaling the average annual turnover timeserie to seq_x,
                        'max_turn':maximum turnover of this business unit and department ever made,
                        'min_turn':minimum turnover of this business unit and department ever made,
                        'bu_num':business unit id,
                        'dep':department id,
                        'day_id':date when the element raw_xy[0] was checked (first date of our timeserie),
                        'no_week':array of the year fraction, the element k of the array corresponds to the fraction of the year when
                            raw_xy[k] was checked,
                        'year_array':array of the years, the element k of the array corresponds the year when raw_xy[k] was checked,
                        'output_knn':average annual turnover predicted from business unit features.
                    }
        dic : dic of data, each element associated with the key i correspond to samples[i], it is used for json saving and loading.
        samples_for_nn: list of data preprocessed for knn, element k is processed from samples[k]
                    and results in a dictionnary based on this format :
                    preprocess_dic={
                                'x':torch tensor of samples[k]["seq_x"],
                                'y':torch tensor of samples[k]["seq_y"],
                                'annual_x':torch tensor of samples[k]["annual_slice_x"],
                                'annual_y':torch tensor of samples[k]["annual_slice_x"],
                                'dep':samples[k]["dep"],
                                'max_turn':sample[k]["max_turn"],
                                'min_turn':sample[k]["min_turn"],
                                'id':k
                            }
        dic_knn : dic of trained knn models per department, each element is associated to the department id key. Each element is
            a k-nearestneighbors model that predicts average annual turnover based on business unit features.
        samples_per_dep: dictionnary where each element associated to the number of a departement is a list of preprocessed dic.
                    for dep=D samples_per_dep[D] returns the list of all preprocessed_dic related to department D constructed
                    from samples list.
        dep: dataset department mode, if dep is set to a number of department D that correspond to a key of samples_per_dep
                    then the dataset will only return preprocessed_dic related to department D.
                    ex : If you want to have only samples related to department 73, set dep to 73.
                         If you want to have all samples not regarding the department then set dep to None.
        dataframe: dataframe of annual turnover used to compute samples.
        dataframe_bu: dataframe of business unit data used to compute samples.
        max_lat: maximum latitude of all business units in dataframe_bu.
        min_lat: minimum latitude of all business units in dataframe_bu.
        max_long: maximum longitude of all business units in dataframe_bu.
        min_long: minimum longitude of all business units in dataframe_bu.
        region_idr_cat: array of all unique but_region_idr_region in dataframe_bu used to one-hot encoding.
        zod_idr_cat: array of all unique zod_idr_zone_dgr in dataframe_bu used to one-hot encoding.
        dep_cat: array of all unique department number in dataframe_bu.
    Args:
        -dic_knn : dic of trained knn models per department, each element is associated to the department id key. Each element is
            a k-nearestneighbors model that predicts average annual turnover based on business unit features.
    
    """
    def __init__(self,dic_knn):
        self.samples=[]
        self.dic={}
        self.dic_knn=dic_knn
        self.samples_for_nn= []
        self.samples_per_dep={73:[],88:[],117:[],127:[]}
        self.dep=None
        self.dataframe=None
        self.dataframe_bu=None
        self.max_lat=None
        self.min_lat=None
        self.max_long=None
        self.min_long=None
        self.region_idr_cat=None
        self.zod_idr_cat=None
        self.dep_cat=None
        
    
    def load_from_df(self, dataframe_turnover, data_frame_bu):
        """
        Load dataset from 2 given dataframes into samples and dic. Each element is a dictionnary on this format:
                dic={
                        'seq_x': timeserie of the last 16 weeks turnovers (of a given business unit and department) 
                            normalized by the maximum and minimum turnovers of this business unit and department ever made,
                        'raw_x': timeserie of the last 16 weeks turnovers (of a given business unit and department),
                        'seq_y':timeserie of the next 8 weeks turnovers (of a given business unit and department) 
                            normalized by the maximum and minimum turnovers of this business unit and department ever made,
                        'raw_y':timeserie of the next 8 weeks turnovers (of a given business unit and department),
                        'raw_xy':timeserie of the last 16 weeks and the next 8 weeks turnovers (of a given business unit and department),
                        'annual_slice_x':timeserie of the last 16 weeks average annual turnovers (of a given business unit and department)
                            scaled to seq_x data,
                        'annual_slice_y':timeserie of the next 8 weeks average annual turnovers (of a given business unit and department)
                            scaled to seq_x data,
                        'scale_factor':scale factor used for scaling the average annual turnover timeserie to seq_x,
                        'max_turn':maximum turnover of this business unit and department ever made,
                        'min_turn':minimum turnover of this business unit and department ever made,
                        'bu_num':business unit id,
                        'dep':department id,
                        'day_id':date when the element raw_xy[0] was checked (first date of our timeserie),
                        'no_week':array of the year fraction, the element k of the array corresponds to the fraction of the year when
                            raw_xy[k] was checked,
                        'year_array':array of the years, the element k of the array corresponds the year when raw_xy[k] was checked,
                        'output_knn':average annual turnover predicted from business unit features.
                    }
            Note that for each timeserie, the index n correspond to a date that is before the index n+1 corresponding date.
        Args:
            - dataframe_turnover(pd.Dataframe) :  a dataframe containing columns : ["day_id],["but_num_business_unit"],
                ["dpt_num_department"],["turnover"]. Its data will be used for computation of annual average turnover.
            - dataframe_bu(pd.Dataframe) :  a dataframe containing columns : ["but_num_business_unit"],["but_latitude"],
                ["but_longitude"],["but_region_idr_region"],["zod_idr_zone_dgr"]. 
                Its data will be used for computation of annual average turnover.         
        """
        self.dep=None
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
        i=0
        self.dic={}
        for bu in tqdm.tqdm(bu_num_list): # ITERATE ON ALL STORES
            dep_list=self.dataframe[(self.dataframe["but_num_business_unit"]==bu)]["dpt_num_department"].unique()
            bu_info=self.dataframe_bu[self.dataframe_bu["but_num_business_unit"]==bu].iloc[0]
            region_idr=bu_info.but_region_idr_region
            zod_idr=bu_info.zod_idr_zone_dgr
            region_idr_enc=utils.map_category_to_vector(self.region_idr_cat,region_idr) #One-hot encoding
            zod_idr_enc=utils.map_category_to_vector(self.zod_idr_cat,zod_idr) #One-hot encoding
            lat=bu_info.but_latitude
            lat=utils.norm(lat,self.max_lat,self.min_lat) #Normalizing
            long=bu_info.but_longitude
            long=utils.norm(long,self.max_long,self.min_long) #Normalizing
            for dep in dep_list: # ITERATE ON ALL DEPARTMENT OF THE STORE
                input_knn=np.concatenate([zod_idr_enc,region_idr_enc,np.array([lat,long])]).reshape(1,-1) #Creates the input feature vector for knn
                output_knn=self.dic_knn[dep].predict(input_knn).reshape(-1) # Predicts the average annual turnover timeserie
                x_old=np.linspace(0,1,num=128)
                x_53=np.linspace(0,1,num=53)
                x_52=np.linspace(0,1,num=52)
                y_53=np.interp(x_53,x_old,output_knn) #Interpolate the average annual turnover timeserie to 53 points (week/year fraction)
                y_52=np.interp(x_52,x_old,output_knn) #Interpolate the average annual turnover timeserie to 52 points (week/year fraction)
                temp_df=self.dataframe[(self.dataframe["but_num_business_unit"]==bu) &
                                (self.dataframe["dpt_num_department"]==dep)]
                max_turnover=np.max(temp_df["turnover"])
                min_turnover=np.min(temp_df["turnover"])
                day_id=np.min(temp_df["day_id"])
                max_day_id=np.max(temp_df["day_id"])-pd.DateOffset(weeks=23)
                while day_id<=max_day_id: # ITERATE ON ALL DATA AVAILABLE FOR THE DEPARTMENT AND STORE
                    end_seq=day_id+pd.DateOffset(weeks=23)
                    sequence=temp_df[(temp_df["day_id"]>=day_id) & (temp_df["day_id"]<=end_seq)].sort_values(by=["day_id"]).reset_index()
                    no_week_array=sequence["week"].to_numpy()
                    year_array=sequence["year"].to_numpy()
                    annual_slice=np.zeros(24)
                    for pos,(w,y) in enumerate(zip(no_week_array,year_array)): # Takes a portion of the average annual turnover corresponding to the period that is used for raw_x and raw_y
                        if y==2015:
                            no_week_array[pos]=x_53[w-1]
                            annual_slice[pos]=y_53[w-1]
                        else:
                            no_week_array[pos]=x_52[w-1]
                            annual_slice[pos]=y_52[w-1]
                    raw_xy=sequence.iloc[0:24]["turnover"].to_numpy()
                    raw_x=raw_xy[0:16] # timeserie of the last 16 weeks turnovers for dep and bu
                    raw_y=raw_xy[16:24] # timeserie of the next 8 weeks turnovers for dep and bu
                    max_x=np.max(raw_x)
                    min_x=np.min(raw_x)
                    annual_slice_x=annual_slice[0:16] # timeserie of the average annual turnover corresponding to the last 16 weeks period
                    annual_slice_y=annual_slice[16:24] # timeserie of the average annual turnover corresponding to the next 8 weeks period
                    normed_raw_x=utils.norm(raw_x,max_x,min_x) #Normalizing
                    annual_slice_y=utils.norm(annual_slice_y,np.max(annual_slice_x),np.min(annual_slice_x)) #Normalizing
                    annual_slice_x=utils.norm(annual_slice_x,np.max(annual_slice_x),np.min(annual_slice_x)) #Normalizing
                    res = op.minimize_scalar(utils.correlation,args=(annual_slice_x,normed_raw_x)) #Find best scale factor
                    scale_factor=res.x
                    annual_slice_x=annual_slice_x*scale_factor #Rescaling to normalized raw_x
                    annual_slice_y=annual_slice_y*scale_factor #Rescaling to normalized raw_x
                    annual_slice_x=utils.inversenorm(annual_slice_x,max_x,min_x)
                    annual_slice_y=utils.inversenorm(annual_slice_y,max_x,min_x)
                    annual_slice_x=utils.norm(annual_slice_x,max_turnover,min_turnover)
                    annual_slice_y=utils.norm(annual_slice_y,max_turnover,min_turnover)
                    seq_x=utils.norm(raw_x,max_turnover,min_turnover)
                    seq_y=utils.norm(raw_y,max_turnover,min_turnover)
                    
                    temp_dic={
                        'seq_x':seq_x,
                        'raw_x':raw_x,
                        'seq_y':seq_y,
                        'raw_y':raw_y,
                        'raw_xy':raw_xy,
                        'annual_slice_x':annual_slice_x,
                        'annual_slice_y':annual_slice_y,
                        'scale_factor':scale_factor,
                        'max_turn':max_turnover,
                        'min_turn':min_turnover,
                        'bu_num':bu,
                        'dep':dep,
                        'day_id':day_id,
                        'no_week':no_week_array,
                        'year_array':year_array,
                        'output_knn':output_knn
                    }
                    self.samples.append(temp_dic.copy())
                    self.dic[i]=temp_dic.copy()
                    i+=1
                    day_id=day_id+pd.DateOffset(weeks=1) # Go to the next data available 

    def save_ds_to_json(self,file):
        """
        Saves dataset dic attribute into a json file using a custom NpEncoder to handle numpy.
        Args:
            - file: an open json file
        """
        json.dump(self.dic, file,cls=NpEncoder)

    def load_from_json(self,file):
        """
        Loads dataset from a json file into dic and samples attributes.
        Args:
            - file: an open json file
        """
        self.dic=json.load(file)
        for key in self.dic.keys():
            self.samples.append(self.dic[key])

    def set_data_for_training(self,free_sample_memory):
        """
        Preprocess the data from samples and stores it into samples_for_knn and samples_per_dep attributes.
        Args:
            - free_sample_memory(boolean) : if True the samples attribute will be set to an empty list.
        
        """
        self.samples_per_dep={73:[],88:[],117:[],127:[]}
        self.samples_for_nn=[]
        for id_s,sample in enumerate(self.samples):
            x=torch.tensor(sample["seq_x"])
            y=torch.tensor(sample["seq_y"])
            annual_x=torch.tensor(sample["annual_slice_x"])
            annual_y=torch.tensor(sample["annual_slice_y"])
            dep=sample["dep"]
            temp_dic={
                                'x':x,
                                'y':y,
                                'annual_x':annual_x,
                                'annual_y':annual_y,
                                'dep':dep,
                                'max_turn':sample["max_turn"],
                                'min_turn':sample["min_turn"],
                                'id':id_s
                            }
            self.samples_for_nn.append(temp_dic.copy())
            self.samples_per_dep[dep].append(temp_dic.copy())
        if free_sample_memory:
            self.samples=[]

    def __len__(self):
        if self.dep in [73,88,117,127]:
            return len(self.samples_per_dep[self.dep])
        else:
            return len(self.samples_for_nn)


    def __getitem__(self, id):
        if self.dep in [73,88,117,127]:
            return self.samples_per_dep[self.dep][id]
        else:
            return self.samples_for_nn[id]