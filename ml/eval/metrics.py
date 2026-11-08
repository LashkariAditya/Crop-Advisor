"""Classification and regression metrics without third-party ML packages."""
import numpy as np

def classification_metrics(y_true,y_pred,n_classes=None):
    labels=np.arange(n_classes or int(max(np.max(y_true),np.max(y_pred))+1)); cm=np.zeros((len(labels),len(labels)),dtype=int)
    for a,b in zip(y_true,y_pred): cm[int(a),int(b)]+=1
    per=[]
    for i in labels:
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; per.append({"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0})
    return {"accuracy":float(np.mean(np.asarray(y_true)==np.asarray(y_pred))),"classes":per,"confusion_matrix":cm.tolist()}

def regression_metrics(y_true,y_pred):
    err=np.asarray(y_pred).reshape(-1)-np.asarray(y_true).reshape(-1); denom=np.maximum(np.abs(y_true),1e-8)
    return {"mae":float(np.mean(np.abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),"mape_pct":float(np.mean(np.abs(err)/denom)*100)}
