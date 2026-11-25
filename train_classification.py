import json
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import CLASS_NAMES,PREDICTORS,load_real_data,prepare_data

CLASSIFICATION_FEATURES=[
 "PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM",
 "hour_sin","hour_cos","month_sin","month_cos",
]

ROOT=__import__("pathlib").Path(__file__).resolve().parent
d=prepare_data(load_real_data())
a=int(len(d)*.70); b=int(len(d)*.85)
train,validation,test=d.iloc[:a],d.iloc[a:b],d.iloc[b:]
model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
    LogisticRegression(class_weight="balanced",max_iter=1200,random_state=2026))
model.fit(train[CLASSIFICATION_FEATURES],train["target_class"])
vp=model.predict(validation[CLASSIFICATION_FEATURES])
v_f1=f1_score(validation["target_class"],vp,average="macro")
tp=model.predict(test[CLASSIFICATION_FEATURES])
metrics={"accuracy":float(accuracy_score(test["target_class"],tp)),
 "balanced_accuracy":float(balanced_accuracy_score(test["target_class"],tp)),
 "macro_f1":float(f1_score(test["target_class"],tp,average="macro"))}
DIRECT_FEATURES=[
 "PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM",
]
training_medians=train[DIRECT_FEATURES].median(numeric_only=True).to_dict()
bundle={"model":model,"features":CLASSIFICATION_FEATURES,"target":"PM2.5 class at the following hour",
"classes":CLASS_NAMES,"thresholds":{"low":12.0,"elevated":35.4,"high":55.4},
"training_medians":training_medians,
"training_median_features":DIRECT_FEATURES,
"uses_lags":False,"uses_rolling_means":False,
"validation_macro_f1":v_f1,"test_metrics":metrics,
"confusion_matrix":confusion_matrix(test["target_class"],tp,labels=CLASS_NAMES).tolist(),
"splits":{"train":len(train),"validation":len(validation),"test":len(test),
"test_start":str(test["timestamp"].iloc[0]),"test_end":str(test["timestamp"].iloc[-1])},
"leakage_control":"Features use only current/past observations and time signals. The target is next-hour PM2.5, and fitting never crosses the chronological split."}
joblib.dump(bundle,ROOT/"results"/"classification_model.joblib")
evaluation={"source":"UCI Beijing Aotizhongxin","target":bundle["target"],"classes":CLASS_NAMES,
"validation_macro_f1":v_f1,"held_out_test":metrics,"splits":bundle["splits"],
"features":CLASSIFICATION_FEATURES,"training_median_features":DIRECT_FEATURES,
"uses_lags":False,"uses_rolling_means":False,
"leakage_control":bundle["leakage_control"]}
(ROOT/"results"/"classification_metrics.json").write_text(__import__("json").dumps(evaluation,ensure_ascii=False,indent=2),encoding="utf-8")

# Rebuild the forecast model in the active scikit-learn environment.
forecast_features=[
 "PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM",
 "pm25_lag_1","pm25_lag_3","pm25_lag_6","pm25_lag_24",
 "pm25_past_3h_mean","pm25_past_6h_mean","pm25_past_24h_mean",
 "hour_sin","hour_cos","month_sin","month_cos",
]
forecast=HistGradientBoostingRegressor(random_state=2026)
forecast.fit(train[forecast_features],train["target_pm25"])
forecast_test=forecast.predict(test[forecast_features])
forecast_metrics={
 "MAE":float(mean_absolute_error(test["target_pm25"],forecast_test)),
 "RMSE":float(__import__("numpy").sqrt(mean_squared_error(test["target_pm25"],forecast_test))),
 "R2":float(r2_score(test["target_pm25"],forecast_test)),
}
joblib.dump({"model":forecast,"features":forecast_features,"selected":"hist_gradient_boosting",
 "station":"Aotizhongxin","sklearn_version":"1.9.1","numpy_version":"2.5.4",
 "pandas_version":"2.3.3","joblib_version":"1.6.0"},ROOT/"results"/"model.joblib")
forecast_report={
 "source":"https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
 "station":"Aotizhongxin","raw_rows":len(load_real_data()),
 "target":"PM2.5 at forecast origin plus one hour (µg/m³)",
 "splits":{"train":len(train),"validation":len(validation),"test":len(test),
 "test_start":str(test["timestamp"].iloc[0]),"test_end":str(test["timestamp"].iloc[-1])},
 "selected_model":"hist_gradient_boosting",
 "results":{"hist_gradient_boosting":{"test":forecast_metrics}},
 "held_out_test":forecast_metrics,
}
(ROOT/"results"/"metrics.json").write_text(json.dumps(forecast_report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metrics,ensure_ascii=False,indent=2))
