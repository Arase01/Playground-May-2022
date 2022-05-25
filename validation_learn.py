"""
Created on Tue May 24 15:19:48 2022

@author: arasekaito
"""

import gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.calibration import CalibratedClassifierCV,calibration_curve

train_path = "input/train.csv"
test_path = "input/test.csv"
colors = px.colors.qualitative.Prism
temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))

def initialsetting(train_path,test_path):
    
    scaler = StandardScaler()
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    
    uni = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
    uni_test = test.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
    train['f_27'] = uni
    test['f_27'] = uni_test
    
    X = train.drop(["target"],axis=1)
    X_test = test
    
    for df in [X,X_test]:
        df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
        df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
        i_00_01_26 = df.f_00 + df.f_01 + df.f_26
        df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(test),columns=X.columns)
    Y = train['target']
    
    return X,X_test,Y,test

def kfold(model,X,X_test,Y):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    Y_valid, model_val_preds, model_test_preds=[],[],[]
    cal_true, cal_pred=[],[]
    sum_AUC,sum_ACC,best_AUC,best_fold = 0,0,0,0
    feat_importance=pd.DataFrame(index=X.columns)
    
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X,Y)):
      print("\nFold {}".format(fold+1))
      X_train, Y_train = X.iloc[train_idx,:], Y[train_idx]
      X_val, Y_val = X.iloc[val_idx,:], Y[val_idx]
      print("Train shape: {}, {}, Valid shape: {}, {}".format(
           X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))
      
      model.fit(X_train,Y_train,verbose=100)
      model_prob = model.predict_proba(X_val)[:,1]
      Y_valid.append(Y_val)
      model_val_preds.append(model_prob)
      model_test_preds.append(model.predict(X_test))
      feat_importance["Importance_Fold"+str(fold)]=model.feature_importances_
      
      calibrated_model = CalibratedClassifierCV(base_estimator=model,cv="prefit")
      cal_fit = calibrated_model.fit(X_train,Y_train)
      cal_probs = calibrated_model.predict_proba(X_val)[:,1]
      prob_true, prob_pred = calibration_curve(Y_val, cal_probs, n_bins=10)
      cal_true.append(prob_true)
      cal_pred.append(prob_pred)
      auc_score = roc_auc_score(Y_val, model_prob)
      acc_score = model.score(X_val,Y_val)
      sum_AUC += auc_score
      sum_ACC += acc_score
      print("Validation AUC = {:.4f}".format(auc_score))
      print("Validation ACC = {:.4f}".format(acc_score))
      
      if auc_score > best_AUC:
          best_AUC = auc_score
          best_ACC = acc_score
          best_fold = fold
          
      del X_train, Y_train, X_val, Y_val
    
    print("")
    print("Mean AUC = {:.4f}  Mean ACC = {:.4f}".format(sum_AUC/(fold+1),sum_ACC/(fold+1)))
    print("")
    print("Best Fold : {}\n Best AUC = {:.4f}  Best ACC = {:.4f}"
          .format(best_fold+1,best_AUC,best_ACC))
    plot_roc_calibration(Y_valid, model_val_preds, cal_true, cal_pred)
    gc.collect()
    return model_test_preds[best_fold]
      
def plot_roc_calibration(y_val, y_prob, mpv_cal, fop_cal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0,1,11), y=np.linspace(0,1,11), 
                             name='Random Chance',mode='lines',
                             line=dict(color="Black", width=1, dash="dot")))
    for i in range(len(y_val)):
        y = y_val[i]
        prob = y_prob[i]
        fpr, tpr, thresh = roc_curve(y, prob)
        roc_auc = auc(fpr,tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color=colors[::-1][i+6], width=3), 
                                 hovertemplate = 'True positive rate = %{y:.3f}, False positive rate = %{x:.3f}',
                                 name='Fold {} AUC = {:.4f}'.format(i+1,roc_auc)))
    fig.update_layout(template=temp, title="Cross-Validation ROC Curves", 
                      hovermode="x unified", width=600,height=500,
                      xaxis_title='False Positive Rate (1 - Specificity)',
                      yaxis_title='True Positive Rate (Sensitivity)',
                      legend=dict(orientation='v', y=.07, x=1, xanchor="right",
                                  bordercolor="black", borderwidth=.5))
    fig.write_image('output/Cross-Validation ROC Curves.png')
    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0,1,11), y=np.linspace(0,1,11), 
                             name='Perfectly Calibrated',mode='lines',
                             line=dict(color="Black", width=1, dash="dot"),legendgroup=2))
    for i in range(len(mpv_cal)):
        mpv = mpv_cal[i]
        fop = fop_cal[i]
        fig.add_trace(go.Scatter(x=mpv, y=fop, line=dict(color=colors[::-1][i+6], width=3), 
                                 hovertemplate = 'Proportion of Positives = %{y:.3f}, Mean Predicted Probability = %{x:.3f}',
                                 name='Fold {}'.format(i+1),legendgroup=2))
    fig.update_layout(template=temp, title="Probability Calibration Curves", 
                      hovermode="x unified", width=600,height=500,
                      xaxis_title='Mean Predicted Probability',
                      yaxis_title='Proportion of Positives',
                      legend=dict(orientation='v', y=.07, x=1, xanchor="right",
                                  bordercolor="black", borderwidth=.5))
    fig.write_image('output/Probability Calibration Curves.png')

def main():
    X,X_test,Y,test = initialsetting(train_path,test_path)
    
    params={'iterations': 284,
            'depth': 10,
            'learning_rate': 0.24641023327616474,
            'random_strength': 0, 
            'bagging_temperature': 0.21482075013237478,
            'od_type': 'Iter', 
            'od_wait': 31,
            'task_type': "GPU"}   
    
    model = CatBoostClassifier(**params)
    
    result = kfold(model,X,X_test,Y)
    
    output = pd.DataFrame({"id": test.id, "target" : result})
    output.to_csv("output/submission.csv",index=False)

if __name__ == '__main__':
    main()
