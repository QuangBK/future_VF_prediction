import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def get_dataset_df():
    RND_STATE = 14
    PATH = '/home/quang/working/OCT_images/oct_dataset/'
    list_files = glob.glob(PATH + '*/*/*/*.jpg')
    print (len(list_files))

    list_files = [x for x in list_files if len(x.split('/')[-1].split('_')) == 2]
    len(list_files)

    list_ids = []
    lr_list = []
    year_list = []
    month_list = []
    timeline_list = []
    for x in tqdm(list_files):
        id_temp = int(x.split('/')[-4].split('.')[0])
        list_ids.append(id_temp)

        date_temp = x.split('/')[-2]
        year_temp, month_temp = date_temp.split('-')[:2]
        year_list.append(int(year_temp))
        month_list.append(int(month_temp))
        timeline_list.append(int(year_temp)*12 + int(month_temp))

        lr_temp = x.split('/')[-1].split('.')[0].split('_')[1]
        lr_list.append(lr_temp)



    df_files  = pd.DataFrame({'path': list_files, 'id': list_ids, 'year': year_list, 'month': month_list, 'timeline': timeline_list, 'lr': lr_list})

    def create_path_lr(row):
        list_path_temp = row['path'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('_')[0]

    def create_date_string(row):
        list_path_temp = row['path'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2]

    df_files['string_path'] = df_files.apply (lambda row: create_path_lr(row), axis=1)
    df_files['string_date'] = df_files.apply (lambda row: create_date_string(row), axis=1)

    print (len(df_files))
    df_files[:3]

    df_report_type = pd.read_csv('report_OCT.csv')
    def create_path(row):
        list_path_temp = row['file_path'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('.')[0]

    df_report_type['string_path'] = df_report_type.apply (lambda row: create_path(row), axis=1)
    df_report_type[:3]

    df_merge = pd.merge(df_files, df_report_type, on='string_path', how='left')

    df_files_rnfl = df_merge[df_merge['type_report'] == 0]
    df_files_gcipl = df_merge[df_merge['type_report'] == 1]
    print (len(df_files_rnfl), len(df_files_rnfl))

    def create_timeline(row):
        text_date = row['test_date']
        year_temp = int(str(text_date)[:4])
        month_temp = int(str(text_date)[4:])    
        return int(year_temp-2000)*12 + int(month_temp)

    LIST_COL = []
    for i in range(1,77):
        LIST_COL.append('GS_' + str(i))

    df_patients = pd.read_csv('patients_editted.csv')
    df_patients['timeline'] = df_patients.apply (lambda row: create_timeline(row), axis=1)
    df_patients = df_patients.dropna(subset=LIST_COL)
    df_patients = df_patients.drop_duplicates(subset=['patient_id', 'OD_OS', 'test_date'], keep='first', inplace=False)
    print (len(df_patients), len(df_files))
    df_patients[:3]

    df_patients["path_rnfl"] = 'None'
    df_patients["path_gcipl"] = 'None'
    for index, row in tqdm(df_patients.iterrows(), total=len(df_patients)):
        temp_rnfl = df_files_rnfl[(df_files_rnfl['id'] == row['patient_id']) & (df_files_rnfl['lr'] == row['OD_OS'].lower())]
        file_path_rnfl = 'None'
        file_path_gcipl = 'None'
        min_dis = 100
        for index_inner, row_inner in temp_rnfl.iterrows():
            temp_dis = abs(row_inner['timeline']- row['timeline'])
            temp_gcipl = df_files_gcipl[df_files_gcipl['string_date'] == row_inner['string_date']]
            if temp_dis < 7 and temp_dis < min_dis and len(temp_gcipl) > 0:
                min_dis = abs(row_inner['timeline']- row['timeline'])
                file_path_rnfl = row_inner['path']
                file_path_gcipl = temp_gcipl.iloc[0]['path']
        if min_dis < 7:
            df_patients['path_rnfl'][index] = file_path_rnfl
            df_patients['path_gcipl'][index] = file_path_gcipl

    df_all = df_patients[(df_patients['path_rnfl'] != 'None') & (df_patients['path_gcipl'] != 'None')]

    print (len(df_all))
    def create_path_t_angle(row):
        list_path_temp = row['path_rnfl'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('.')[0].split('_')[1]

    df_all['string_path'] = df_all.apply (lambda row: create_path_t_angle(row), axis=1)
    df_all[:3]


    ids_train, ids_test = train_test_split(df_all['patient_id'].unique(), test_size=0.33, random_state=RND_STATE)

    df_train = df_all.loc[df_all['patient_id'].isin(ids_train)]
    df_test = df_all.loc[df_all['patient_id'].isin(ids_test)]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print (len(df_train), len(df_test))

    df_angle = pd.read_csv('./localization_model/pred_pos.csv')

    def create_path_angle(row):
        list_path_temp = row['file'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('.')[0].split('_')[1]

    df_angle['string_path'] = df_angle.apply (lambda row: create_path_angle(row), axis=1)
    df_angle[:3]

    list_angles_F_df = df_angle['angle_od'].to_numpy()
    list_angles_F_df_temp = list_angles_F_df[(list_angles_F_df < 75) | (list_angles_F_df > -10)]
    list_angles_F_df = np.where((list_angles_F_df > 75) | (list_angles_F_df < -10), list_angles_F_df_temp.mean(), list_angles_F_df)
    df_angle['angle_od_refined'] = list_angles_F_df
    
    return df_all, df_train, df_test, df_angle


def get_dataset_df_EDI():
    RND_STATE = 14
    PATH = '/home/quang/working/OCT_images/oct_dataset/'
    list_files = glob.glob(PATH + '*/*/*/*.jpg')
    print (len(list_files))

    list_files = [x for x in list_files if len(x.split('/')[-1].split('_')) == 2]
    len(list_files)

    list_ids = []
    lr_list = []
    year_list = []
    month_list = []
    timeline_list = []
    for x in tqdm(list_files):
        id_temp = int(x.split('/')[-4].split('.')[0])
        list_ids.append(id_temp)

        date_temp = x.split('/')[-2]
        year_temp, month_temp = date_temp.split('-')[:2]
        year_list.append(int(year_temp))
        month_list.append(int(month_temp))
        timeline_list.append(int(year_temp)*12 + int(month_temp))

        lr_temp = x.split('/')[-1].split('.')[0].split('_')[1]
        lr_list.append(lr_temp)



    df_files  = pd.DataFrame({'path': list_files, 'id': list_ids, 'year': year_list, 'month': month_list, 'timeline': timeline_list, 'lr': lr_list})

    def create_path_lr(row):
        list_path_temp = row['path'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('_')[0]

    def create_date_string(row):
        list_path_temp = row['path'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2]

    df_files['string_path'] = df_files.apply (lambda row: create_path_lr(row), axis=1)
    df_files['string_date'] = df_files.apply (lambda row: create_date_string(row), axis=1)

    print (len(df_files))
    df_files[:3]

    df_report_type = pd.read_csv('report_OCT.csv')
    def create_path(row):
        list_path_temp = row['file_path'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('.')[0]

    df_report_type['string_path'] = df_report_type.apply (lambda row: create_path(row), axis=1)
    df_report_type[:3]

    df_merge = pd.merge(df_files, df_report_type, on='string_path', how='left')

    df_files_rnfl = df_merge[df_merge['type_report'] == 0]
    df_files_gcipl = df_merge[df_merge['type_report'] == 1]
    print (len(df_files_rnfl), len(df_files_rnfl))

    def create_timeline(row):
        text_date = row['test_date']
        year_temp = int(str(text_date)[:4])
        month_temp = int(str(text_date)[4:])    
        return int(year_temp-2000)*12 + int(month_temp)

    LIST_COL = []
    for i in range(1,77):
        LIST_COL.append('GS_' + str(i))

    df_patients = pd.read_csv('patients_editted.csv')
    df_patients['timeline'] = df_patients.apply (lambda row: create_timeline(row), axis=1)
    df_patients = df_patients.dropna(subset=LIST_COL)
    df_patients = df_patients.drop_duplicates(subset=['patient_id', 'OD_OS', 'test_date'], keep='first', inplace=False)
    print (len(df_patients), len(df_files))
    df_patients[:3]

    list_folders = glob.glob('/home/quang/working/OCT_images/original_OCT_dataset/*/')


    list_EDI = []
    list_lr_EDI = []
    list_path_EDI = []
    for path_temp in list_folders:
        id_temp = path_temp.split('/')[-2]
        list_files_temp = glob.glob(path_temp + id_temp + '*/*.jpg')
        if len(list_files_temp) > 0:
            list_EDI.append(int(id_temp.split('.')[0]))
            list_lr_EDI.append('OD' if id_temp.split('.')[1][0] == 'R' else 'OS')
            list_path_EDI.append(list_files_temp[0])
        
    len(list_EDI)
    df_EDI = pd.DataFrame({'id':list_EDI, 'OD_OS':list_lr_EDI, 'path': list_path_EDI})

    df_patients["path_rnfl"] = 'None'
    df_patients["path_gcipl"] = 'None'
    df_patients["path_EDI"] = 'None'
    for index, row in tqdm(df_patients.iterrows(), total=len(df_patients)):
        temp_rnfl = df_files_rnfl[(df_files_rnfl['id'] == row['patient_id']) & (df_files_rnfl['lr'] == row['OD_OS'].lower())]
        file_path_rnfl = 'None'
        file_path_gcipl = 'None'
        min_dis = 100
        for index_inner, row_inner in temp_rnfl.iterrows():
            temp_dis = abs(row_inner['timeline']- row['timeline'])
            temp_gcipl = df_files_gcipl[df_files_gcipl['string_date'] == row_inner['string_date']]
            if temp_dis < 7 and temp_dis < min_dis and len(temp_gcipl) > 0:
                min_dis = abs(row_inner['timeline']- row['timeline'])
                file_path_rnfl = row_inner['path']
                file_path_gcipl = temp_gcipl.iloc[0]['path']
        if min_dis < 7:
            df_patients['path_rnfl'][index] = file_path_rnfl
            df_patients['path_gcipl'][index] = file_path_gcipl
        if len(df_EDI[(df_EDI['id'] == row['patient_id']) & (df_EDI['OD_OS'] == row['OD_OS'])]) > 0:
            df_patients['path_EDI'][index] = df_EDI[(df_EDI['id'] == row['patient_id']) & (df_EDI['OD_OS'] == row['OD_OS'])].iloc[0]['path'] 

    df_all = df_patients[(df_patients['path_rnfl'] != 'None') & (df_patients['path_gcipl'] != 'None') & (df_patients['path_EDI'] != 'None')]

    print (len(df_all))
    def create_path_t_angle(row):
        list_path_temp = row['path_rnfl'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('.')[0].split('_')[1]

    df_all['string_path'] = df_all.apply (lambda row: create_path_t_angle(row), axis=1)
    df_all[:3]


    ids_train, ids_test = train_test_split(df_all['patient_id'].unique(), test_size=0.33, random_state=RND_STATE)

    df_train = df_all.loc[df_all['patient_id'].isin(ids_train)]
    df_test = df_all.loc[df_all['patient_id'].isin(ids_test)]

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print (len(df_train), len(df_test))

    df_angle = pd.read_csv('./localization_model/pred_pos.csv')

    def create_path_angle(row):
        list_path_temp = row['file'].split('/')    
        return list_path_temp[-4].split('.')[0] + '_' + list_path_temp[-2] + '_' + list_path_temp[-1].split('.')[0].split('_')[1]

    df_angle['string_path'] = df_angle.apply (lambda row: create_path_angle(row), axis=1)
    df_angle[:3]

    list_angles_F_df = df_angle['angle_od'].to_numpy()
    list_angles_F_df_temp = list_angles_F_df[(list_angles_F_df < 75) | (list_angles_F_df > -10)]
    list_angles_F_df = np.where((list_angles_F_df > 75) | (list_angles_F_df < -10), list_angles_F_df_temp.mean(), list_angles_F_df)
    df_angle['angle_od_refined'] = list_angles_F_df
    
    return df_all, df_train, df_test, df_angle

