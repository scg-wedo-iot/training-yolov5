import pandas as pd
from sklearn.metrics import accuracy_score


OCR_predict = pd.read_csv("data_scg_test/OCR_predict_640.csv")
label_ocr_WEDO_internal = pd.read_csv("data_scg_test/label_ocr_WEDO_internal.csv")


df = OCR_predict.merge(label_ocr_WEDO_internal, how='inner', on='name_img')

y_true = df["ocr_ref"].astype('int')
y_pred = df["OCR_predict"].astype('int')
raw_accuracy= round(accuracy_score(y_true, y_pred)*100,2)
print(f"RAW = {raw_accuracy}")

y_true_remove_last_digi = (y_true/10).astype('int')
y_pred_remove_last_digi = (y_pred/10).astype('int')
cut_accuracy = round(accuracy_score(y_true_remove_last_digi, y_pred_remove_last_digi)*100,2)
print(f"Cut = {cut_accuracy}")

#