from __future__ import division

# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from scipy.stats.stats import pearsonr
from math import sqrt
import datetime
from dateutil.rrule import *
from datetime import date
import matplotlib.pyplot as plt
import openpyxl
import xlrd

# Read recipe inputs

input_dataset = dataiku.Dataset(get_input_names_for_role('input_dataset')[0])
input_df = input_dataset.get_dataframe()

metrics_folder_path = dataiku.Folder(get_output_names_for_role('metrics_folder')[0]).get_path()

resource_path = get_recipe_resource()

book = openpyxl.load_workbook(resource_path + "/confusion_matrix_TEMPLATE.xlsx")
sheet = book.worksheets[0]



#Get metrics from training
training_metrics_list = dataiku.Model(get_input_names_for_role('trained_model')[0])
training_metrics_list.get_predictor()
for version in training_metrics_list.versions:
    if version['active']==True:
        training_metrics = version
training_metrics = training_metrics['snippet']
if training_metrics['trainInfo']['kfold'] == True:
    accuracystd = training_metrics['accuracystd']
    recallstd = training_metrics['recallstd']
    precisionstd = training_metrics['precisionstd']

prediction_type = get_recipe_config()['prediction_type']

now = datetime.datetime.now()


def get_confusion_matrix(df, y_actual, y_pred):
    """
    Parameters
    ----------
    df : pandas dataframe
        the dataframe containing columns with predicted and actual class values
    y_actual : string
        name of the column of actual class values
    y_pred : string
        name of the column of predicted class values, based on some model

    Returns
    -------
    TP : int
        number of true positives
    FP : int
        number of false positives
    TN : int
        number of true negatives
    FN : int
        number of false negatives
    """
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for index, row in df.iterrows():

        if row[y_actual] == row[y_pred] == 1:
            TP += 1
        if row[y_pred] == 1 and row[y_actual] != row[y_pred]:
            FP += 1
        if row[y_actual] == row[y_pred] == 0:
            TN += 1
        if row[y_pred] == 0 and row[y_actual] != row[y_pred]:
            FN += 1

    return(TP, FP, TN, FN)


def get_classification_metrics(df, y_actual, y_pred_prob, y_pred):
    """
    Parameters
    ----------
    df : pandas dataframe
        the dataframe containing columns with predicted and actual class values
    y_actual : string
        name of the column of actual class values
    y_pred_prob : string
        name of the column of predicted class probability values, based on some model
    y_pred : string
        name of the column of predicted class values, based on some model
    
    Returns
    -------
    precision : double
    recall : double
    accuracy : double
    roc_auc : double
    f1 : double
    hamming_loss_score : double
    log_loss_score : double
    lift : double
    """
    
    true_positives, false_positives, true_negatives, false_negatives = get_confusion_matrix(df, y_actual, y_pred)
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 'NA'
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 'NA'
    try:
        accuracy = (true_positives + true_negatives)/(true_positives + false_positives + true_negatives + false_negatives)
    except:
        accuracy = 'NA'
    try:
        roc_auc = roc_auc_score(df[y_actual], df[y_pred_prob])
    except:
        roc_auc = 'NA'
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1='NA'
    try:
        hamming_loss_score = hamming_loss(df[y_actual], df[y_pred])
    except:
        hamming_loss_score = 'NA'
    try:
        log_loss_score = log_loss(df[y_actual], df[y_pred_prob])
    except:
        log_loss_score = 'NA'
    try:
        lift = (true_positives/(true_positives+false_negatives))/((true_positives+false_positives)/(true_positives+true_negatives+false_positives+false_negatives))
    except:
        lift = 'NA'
    try:
        mcc = matthews_corrcoef(df[y_actual], df[y_pred])
    except:
        mcc = 'NA'
    total_records = true_positives + false_positives + true_negatives + false_negatives
    
    # Edit Confusion Matrix Excel File
    sheet["D4"] = true_positives
    sheet["E4"] = false_positives
    sheet["F4"] = true_positives + false_positives
    
    sheet["D5"] = false_negatives
    sheet["E5"] = true_negatives
    sheet["F5"] = true_negatives + false_negatives
    
    sheet["D6"] = true_positives + false_negatives
    sheet["E6"] = true_negatives + false_positives
    sheet["F6"] = total_records
    
    sheet["D10"] = false_positives/total_records
    sheet["D11"] = false_negatives/total_records
    sheet["D12"] = true_positives/total_records
    sheet["D13"] = true_positives/(true_positives + false_positives)
    sheet["D14"] = mcc
    sheet["E13"] = "(" + str(round( (true_positives/(true_positives + false_positives) - precisionstd)*100,1) ) + "% , " + str(round( (true_positives/(true_positives + false_positives) + precisionstd)*100,1) ) + "%)"
    
    book.save(metrics_folder_path + '/confusion_matrix' + str(now) + '.xlsx')
    
    
    # Build ROC Curbve
    fpr = dict()
    tpr = dict()
    for i in range(df.shape[1]):
         fpr[i], tpr[i], _ = roc_curve(df[y_actual], df[y_pred])
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(metrics_folder_path + '/roc_curve' + str(now) + '.pdf')
    
    
    return precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records

def get_classification_metrics_limited(df, y_actual, y_pred):
    """
    Parameters
    ----------
    df : pandas dataframe
        the dataframe containing columns with predicted and actual class values
    y_actual : string
        name of the column of actual class values
    y_pred : string
        name of the column of predicted class values, based on some model
    
    Returns
    -------
    precision : double
    recall : double
    accuracy : double
    f1 : double
    hamming_loss_score : double
    lift : double
    """
    
    true_positives, false_positives, true_negatives, false_negatives = get_confusion_matrix(df, y_actual, y_pred)
    
    try:
        precision = true_positives/(true_positives + false_positives)
    except:
        precision = 'NA'
    try:
        recall = true_positives/(true_positives + false_negatives)
    except:
        recall = 'NA'
    try:
        accuracy = (true_positives + true_negatives)/(true_positives + false_positives + true_negatives + false_negatives)
    except:
        accuracy = 'NA'
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1='NA'
    try:
        hamming_loss_score = hamming_loss(df[y_actual], df[y_pred])
    except:
        hamming_loss_score = 'NA'
    
    try:
        lift = (true_positives/(true_positives+false_negatives))/((true_positives+false_positives)/(true_positives+true_negatives+false_positives+false_negatives))
    except:
        lift='NA'
    try:
        mcc = matthews_corrcoef(df[y_actual], df[y_pred])
    except:
        mcc = 'NA'
    total_records = true_positives + false_positives + true_negatives + false_negatives
    
    sheet["D4"] = true_positives
    sheet["E4"] = false_positives
    sheet["F4"] = true_positives + false_positives
    
    sheet["D5"] = false_negatives
    sheet["E5"] = true_negatives
    sheet["F5"] = true_negatives + false_negatives
    
    sheet["D6"] = true_positives + false_negatives
    sheet["E6"] = true_negatives + false_positives
    sheet["F6"] = total_records
    
    sheet["D10"] = false_positives/total_records
    sheet["D11"] = false_negatives/total_records
    sheet["D12"] = true_positives/total_records
    sheet["D13"] = true_positives/(true_positives + false_positives)
    sheet["D14"] = mcc
    sheet["E13"] = "(" + str(round( (true_positives/(true_positives + false_positives) - precisionstd)*100,1) ) + "% , " + str(round( (true_positives/(true_positives + false_positives) + precisionstd)*100,1) ) + "%)"
    
    book.save(metrics_folder_path + '/confusion_matrix' + str(now) + '.xlsx')
    
    return precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records


def get_regression_metrics(df, y_actual, y_pred):
    """
    Parameters
    ----------
    df : pandas dataframe
        the dataframe containing columns with predicted and actual class values
    y_actual : string
        name of the column of actual numeric values
    y_pred : string
        name of the column of predicted numeric values, based on some model
    
    Returns
    -------
    evs : double
    mae : double
    mse : double
    msle : double
    rmse : double
    rmsle : double
    r2 : double
    pearson_coef : double
    pearson_p_val : double
    """
    
    evs = explained_variance_score(df[y_actual], df[y_pred])
    mae = mean_absolute_error(df[y_actual], df[y_pred])
    mse = mean_squared_error(df[y_actual], df[y_pred])
    msle = mean_squared_log_error(df[y_actual], df[y_pred])
    rmse = sqrt(mse)
    rmsle = sqrt(msle)
    r2 = r2_score(df[y_actual], df[y_pred])
    pearson_coef, pearson_p_val = pearsonr(df[y_actual], df[y_pred])

    return evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val



    


target_actual_col = get_recipe_config().get('target_actual_col', None)

model_drift_time = get_recipe_config().get('model_drift_time', None)

if model_drift_time != 'none':
    model_drift_time_col = get_recipe_config().get('model_drift_time_col', None)
    start_date = min(input_df[model_drift_time_col])
    end_date = max(input_df[model_drift_time_col])
    
    if model_drift_time == 'monthly':
        dates_list = [day.isoformat() for day in rrule(MONTHLY, bymonthday=1, dtstart=start_date, until=end_date)]
    elif model_drift_time == 'yearly':
        dates_list = [day.isoformat() for day in rrule(YEARLY, byyearday=1, dtstart=start_date.replace(month=1, day=1) , until=end_date)]
    elif model_drift_time == 'weekly':
        dates_list = [day.isoformat() for day in rrule(WEEKLY, byweekday=MO, dtstart=start_date, until=end_date)]
    elif model_drift_time == 'daily':
        dates_list = [day.isoformat() for day in rrule(DAILY, dtstart=start_date, until=end_date)]

categorical_breakdown_col = get_recipe_config().get('categorical_breakdown_col', None)

# If user chooses classification, calculate classification metrics
if prediction_type == 'classification':
    
    classification_predictions_or_probabilities = get_recipe_config().get('classification_predictions_or_probabilities', None)
    
    if classification_predictions_or_probabilities == 'probabilities':
        prediction_probability_col = get_recipe_config().get('prediction_probability_col', None)
        probability_threshold = get_recipe_config()['probability_threshold']

        if 'prediction_class' in input_df.columns:
            prediction_class_col = 'prediction_class_calculated'
        else:
            prediction_class_col = 'prediction_class'

        # Create predicted classes column based on the predicted probabilities column
        input_df[prediction_class_col] = [1 if float(x) >= float(probability_threshold) else 0 for x in input_df[prediction_probability_col]]

        # Convert the actual target column to integers (if coded as 1.0, 0.0)
        input_df[target_actual_col] = input_df[target_actual_col].astype(int)
        if model_drift_time != 'none':
            columns_list = ['time_computed', 'eval_period_start', 'eval_period_end', 'probability_threshold', 'precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'hamming_loss', 'log_loss', 'lift', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'total_records']
            master_metrics_df = pd.DataFrame(columns=columns_list)
            
            prev_date_val = None
            for date_val in dates_list:
                
                if prev_date_val:
                    filtered_df = input_df[(input_df[model_drift_time_col] <= date_val) & (input_df[model_drift_time_col] > prev_date_val)]
                else:
                    prev_date_val = start_date
                    filtered_df = input_df[input_df[model_drift_time_col] <= date_val]

                precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records = get_classification_metrics(filtered_df, target_actual_col, prediction_probability_col, prediction_class_col)

                metrics_df = pd.DataFrame([[now, prev_date_val, date_val, probability_threshold, precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records]], columns=columns_list)
                master_metrics_df = master_metrics_df.append(metrics_df)

                # Set previous date value for next iteration equal to this current value
                prev_date_val = date_val

        elif categorical_breakdown_col: 
            columns_list = ['time_computed', categorical_breakdown_col, 'probability_threshold', 'precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'hamming_loss', 'log_loss', 'lift', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'total_records']
            master_metrics_df = pd.DataFrame(columns=columns_list)
            
            for category in input_df[categorical_breakdown_col].unique():
                
                filtered_df = input_df[input_df[categorical_breakdown_col]==category]

                precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records = get_classification_metrics(filtered_df, target_actual_col, prediction_probability_col, prediction_class_col)

                metrics_df = pd.DataFrame([[now, category, probability_threshold, precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records]], columns=columns_list)
                master_metrics_df = master_metrics_df.append(metrics_df)
            
        else:
            precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records = get_classification_metrics(input_df, target_actual_col, prediction_probability_col, prediction_class_col)

            columns_list = ['time', 'probability_threshold', 'precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'hamming_loss', 'log_loss', 'lift', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'total_records']
            master_metrics_df = pd.DataFrame([[now, probability_threshold, precision, recall, accuracy, roc_auc, f1, hamming_loss_score, log_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records]], columns=columns_list)
    
    else:
        prediction_class_col = get_recipe_config().get('prediction_class_col', None)
        
        # Convert the actual target column to integers (if coded as 1.0, 0.0)
        input_df[target_actual_col] = input_df[target_actual_col].astype(int)
        
        if model_drift_time != 'none':
            columns_list = ['time_computed', 'eval_period_start', 'eval_period_end', 'precision', 'recall', 'accuracy', 'f1', 'hamming_loss', 'lift', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'total_records']
            master_metrics_df = pd.DataFrame(columns=columns_list)
            
            prev_date_val = None
            for date_val in dates_list:
                
                if prev_date_val:
                    filtered_df = input_df[(input_df[model_drift_time_col] <= date_val) & (input_df[model_drift_time_col] > prev_date_val)]
                else:
                    prev_date_val = start_date
                    filtered_df = input_df[input_df[model_drift_time_col] <= date_val]
                    
                precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records =  get_classification_metrics_limited(filtered_df, target_actual_col, prediction_class_col)

                metrics_df = pd.DataFrame([[now, prev_date_val, date_val, precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records]], columns=columns_list)
                master_metrics_df = master_metrics_df.append(metrics_df)

                # Set previous date value for next iteration equal to this current value
                prev_date_val = date_val
                
        elif categorical_breakdown_col:
            columns_list = ['time_computed', categorical_breakdown_col, 'precision', 'recall', 'accuracy', 'f1', 'hamming_loss', 'lift', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'total_records']
            master_metrics_df = pd.DataFrame(columns=columns_list)
            
            for category in input_df[categorical_breakdown_col].unique():
                
                filtered_df = input_df[input_df[categorical_breakdown_col]==category]
                    
                precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records =  get_classification_metrics_limited(filtered_df, target_actual_col, prediction_class_col)

                metrics_df = pd.DataFrame([[now, category, precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records]], columns=columns_list)
                master_metrics_df = master_metrics_df.append(metrics_df)
        
        else:
            precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records =  get_classification_metrics_limited(input_df, target_actual_col, prediction_class_col)

            columns_list = ['time', 'precision', 'recall', 'accuracy', 'f1', 'hamming_loss', 'lift', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'total_records']
            master_metrics_df = pd.DataFrame([[now, precision, recall, accuracy, f1, hamming_loss_score, lift, true_positives, false_positives, true_negatives, false_negatives, total_records]], columns=columns_list)
    
        
# If user chooses regression, calculate regression metrics
else:

    prediction_col = get_recipe_config().get('prediction_col', None)
    
    if model_drift_time != 'none':
            columns_list = ['time_computed', 'eval_period_start', 'eval_period_end', 'evs', 'mae', 'mse', 'msle', 'rmse', 'rmsle', 'r2', 'pearson_coef', 'pearson_p_val']
            master_metrics_df = pd.DataFrame(columns=columns_list)
            
            prev_date_val = None
            
            for date_val in dates_list:
                
                if prev_date_val:
                    filtered_df = input_df[(input_df[model_drift_time_col] <= date_val) & (input_df[model_drift_time_col] > prev_date_val)]
                else:
                    prev_date_val = start_date
                    filtered_df = input_df[input_df[model_drift_time_col] <= date_val]
                    
                evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val = get_regression_metrics(filtered_df, target_actual_col, prediction_col)


                metrics_df = pd.DataFrame([[now, prev_date_val, date_val, evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val]], columns=columns_list)
                master_metrics_df = master_metrics_df.append(metrics_df)

                # Set previous date value for next iteration equal to this current value
                prev_date_val = date_val

    elif categorical_breakdown_col:
        columns_list = ['time_computed', categorical_breakdown_col, 'evs', 'mae', 'mse', 'msle', 'rmse', 'rmsle', 'r2', 'pearson_coef', 'pearson_p_val']
        master_metrics_df = pd.DataFrame(columns=columns_list)

        for category in input_df[categorical_breakdown_col].unique():

            filtered_df = input_df[input_df[categorical_breakdown_col]==category]

            evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val = get_regression_metrics(filtered_df,target_actual_col, prediction_col)

            metrics_df = pd.DataFrame([[now, category, evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val]], columns=columns_list)
            master_metrics_df = master_metrics_df.append(metrics_df)                
                
    else:
        evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val = get_regression_metrics(input_df,target_actual_col, prediction_col)

        columns_list = ['time', 'evs', 'mae', 'mse', 'msle', 'rmse', 'rmsle', 'r2', 'pearson_coef', 'pearson_p_val']
        metrics_df = pd.DataFrame([[now, evs, mae, mse, msle, rmse, rmsle, r2, pearson_coef, pearson_p_val]], columns=columns_list)





# Write recipe outputs
metrics_dataset = dataiku.Dataset(get_output_names_for_role('metrics')[0])
metrics_dataset.write_with_schema(master_metrics_df)



