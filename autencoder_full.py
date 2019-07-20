import os
import pandas as pd
import path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
'''                     feature selection
from CFS import cfs, merit_calculation
from sklear
n.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
pca = PCA(50)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
'''
# ----------------------------------------- PATH AND FILE NAMING -----------------------------------------
files_name = ["device_category_training_SAMPLE.csv", "device_category_validation_SAMPLE.csv",
              "device_category_test_SAMPLE.csv"]
out_path = r"C:\Users\Administrator\Desktop\DATA"
model_path = r"C:\Users\Administrator\Desktop\models"
raw_path = r"C:\Users\Administrator\Desktop\DATA\RAW_DATA"
# ----------------------------------------- END PATH AND FILE NAMING -----------------------------------------


# --------------------------------------- DATA STRUCTURE BUILD AND GLOBAL DECLARATION  -------------------------------

model_exists = False
device_categories = pd.read_csv(os.path.join(out_path, "device_categories.csv"), header=None)
device_categories = device_categories.drop_duplicates().values.tolist()
device_categories = [item[0] for item in device_categories]
df_autoencoders = pd.DataFrame(index=device_categories, columns=['autoencoder', 'th', 'feats', 'model'])
ls = []
for element in device_categories:
    ls.append(str(element) + "_0")
    ls.append(str(element) + "_1")
df_results1 = pd.DataFrame(index=ls,
                           columns=["tst_0", "tst_1", "trn_0", "trn_1", "vld_0", "vld_1", "th", 'tst_Total_Session',
                                    'tst_Precision', 'tst_TPR', 'tst_FPR', 'tst_F1', 'tst_ROC_AUC',
                                    'train_Total_Session', 'train_Precision', 'train_TPR', 'val_Total_Session',
                                    'val_Precision', 'val_TPR', 'tp_tr', 'fp_tr', 'tn_tr', 'fn_tr',
                                    'tp_val', 'fp_val', 'tn_val', 'fn_val', 'tp', 'fp', 'tn', 'fn'])
df_results_majority = pd.DataFrame(index=ls,
                                   columns=["tst_0", "tst_1", "th", 'tst_Total_Session', 'tst_Precision', 'tst_TPR',
                                            'tst_FPR', 'tst_F1', 'tst_ROC_AUC', 'tp', 'fp', 'tn', 'fn'])
df_confusion = pd.DataFrame(index=device_categories, columns=device_categories)  # rows = pred, cols = act
df_confusion[:] = 0


#  --------------------------------------------------- AUTOENCODER CLASS --------------------------------------
class autoencoder:
    def __init__(self):
        self.categories = None
        self.feat_scale = None
        self.error_scale = None
        self.pred_classes = None
        self.model = None
        self.dense1 = None

    def get_saved_model(self, model):
        self.model = model

    def create_model(self, new_training_feat):
        """
        creates the autoencoder model conf.
        :param new_training_feat: training features
        :return: the model conf.
        """
        input_size = new_training_feat.shape[1]
        self.dense1 = input_size
        model = Sequential()
        model.add(Dense(input_dim=input_size, units=int(input_size / 3), activation='relu',
                        activity_regularizer=regularizers.l1(10e-5)))
        model.add(Dense(input_dim=int(input_size / 3), units=int(input_size / 6), activation='relu'))
        model.add(Dense(input_dim=int(input_size / 6), units=int(input_size / 3), activation='relu'))
        model.add(Dense(input_dim=int(input_size / 3), units=int(input_size), activation='relu'))
        model.add(Dense(units=int(input_size / 3), activation='relu'))
        model.add(Dense(input_size, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def train_model(self, model, train_features, valid_features):
        """
        training the model based on train_features and train_labels
        :param model: the model conf.
        :param train_features: the training features
        :param valid_features: the validation features
        :return: the trained models th
        """
        # mc = ModelCheckpoint(f'best_model_{label}.h5', verbose=2, monitor='val_loss', mode='min', save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
        self.feat_scale = MinMaxScaler()
        model.fit(x=train_features, y=train_features, epochs=100,
                  validation_data=(valid_features, valid_features), verbose=1, callbacks=[es])
        th_train = autoencoder.get_prediction(model, valid_features)[0]
        return th_train

    @staticmethod
    def get_prediction(model, features):
        predictions_model = model.predict(features)
        mses = ((np.array(features) - np.array(predictions_model)) ** 2).mean(axis=1).reshape(-1, )
        return determine_th(mses, 95), predictions_model

    @staticmethod
    def plot_result(trained_ae):
        loss = trained_ae.history['loss']
        val_loss = trained_ae.history['val_loss']
        epochs = range(100)
        plt.figure()
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()
#  ---------------------------------------------------END AUTOENCODER CLASS --------------------------------------

#  --------------------------------------------------- PRE PROCESS  --------------------------------------


def create_dictionaries(device):
    train_feat4 = pd.read_csv(os.path.join(out_path, f"train_features_{device}.csv"))
    valid_feat4 = pd.read_csv(os.path.join(out_path, f"valid_features_{device}.csv"))
    train_label4 = pd.read_csv(os.path.join(out_path, f"train_labels_{device}.csv"))
    val_label4 = pd.read_csv(os.path.join(out_path, f"valid_labels_{device}.csv"))
    return train_feat4, valid_feat4, train_label4, val_label4


def separate_feat_label(data):
    """
    separates the data to a features set and label set
    :param data: numPy array of all the features and labels
    :return: tuple of 2 numPy array
    """
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1].ravel()  # .ravel() = to single list
    return features, labels


def pre_process_categories():
    global out_path
    global raw_path
    with open(os.path.join(out_path, "test_features.csv"), "w", newline=''):
        pass
    with open(os.path.join(out_path, "test_labels.csv"), "w", newline=''):
        pass
    pd.DataFrame(device_categories).to_csv("device_categories.csv", "w", header=False, index=False)
    for name in ['train_features', 'train_labels', 'valid_labels', 'valid_features']:
        for dc in device_categories:
            with open(os.path.join(out_path + f"{name}_{dc}.csv"), "w", newline=''):
                pass
    for partition_suffix in files_name:
        if partition_suffix.__contains__("test"):
            filename = path.join(raw_path, partition_suffix)
            chunk_size = 10 ** 7
            check_num = 1
            for chunk in pd.read_csv(filename, chunksize=chunk_size):
                print(check_num)
                check_num = check_num+1
                chunk = chunk.drop(columns=['start', 'start_original'])
                chunk = chunk.loc[:, ~chunk.columns.duplicated()]
                pre_process_categories_file(chunk, partition_suffix)
    return None


def pre_process_categories_file(df_input_data, partition_suffix):
    """
    pre process the data before training the autoencoder on it
    :param df_input_data: the csv file of the input data
    :param partition_suffix: the name of the csv file
    :return: NaN
    """
    global out_path
    if partition_suffix.__contains__("test"):
        test_features, test_labels = separate_feat_label(df_input_data)  # np's for feat and label
        with open(os.path.join(out_path, "test_features.csv"), 'a', newline='') as f:
            pd.DataFrame(test_features).to_csv(f, header=False, index=False)

        with open(os.path.join(out_path, "test_labels.csv"), 'a', newline='') as f:
            pd.DataFrame(test_labels).to_csv(f, header=False, index=False)
    else:
        for dev_class in device_categories:
            df_data_per_category = df_input_data[df_input_data.device_category == dev_class]
            if partition_suffix.__contains__("train"):
                print(f"writing {dev_class}")
                train_features, train_labels = separate_feat_label(df_data_per_category)
                with open(os.path.join(out_path, f"train_features_{dev_class}.csv"), 'a', newline='') as f:
                    pd.DataFrame(train_features).to_csv(f, header=False, index=False)
                with open(os.path.join(out_path, f"train_labels_{dev_class}.csv"), 'a', newline='') as f:
                    pd.DataFrame(train_labels).to_csv(f, header=False, index=False)
            else:
                valid_features, valid_labels = separate_feat_label(df_data_per_category)
                with open(os.path.join(out_path, f"valid_features_{dev_class}.csv"), 'a', newline='') as f:
                    pd.DataFrame(valid_features).to_csv(f, header=False, index=False)
                with open(os.path.join(out_path, f"valid_labels_{dev_class}.csv"), 'a', newline='') as f:
                    pd.DataFrame(valid_labels).to_csv(f, header=False, index=False)
    return


#  ---------------------------------------------------END PRE PROCESS  --------------------------------------

#  -------------------------------------------- EVALUATION & RESULT PROCESS  -----------------------------------
def determine_th(dataset, percentile):
    """
    determine threshold at certain percentile
    :param dataset: the dataset
    :param percentile: the required percentile
    :return:
    """
    th_determined = np.percentile(dataset, percentile)
    return th_determined


def majority_test():
    test_feat['mac'] = test_macs
    test_feat['label'] = test_label
    by_macs = {}
    test_macs2 = list(set(test_macs))
    for mac_add in test_macs2:
        by_macs[mac_add] = test_feat[test_feat['mac'] == mac_add]
    for device in device_categories:
        final_mses = {}
        final_labels = {}
        th_new = df_autoencoders['th'][device]
        for mac_add2 in by_macs:
            test_label2 = by_macs[mac_add2].iloc[:, -1].ravel()
            test_feat2 = feat_scale.fit_transform(by_macs[mac_add2].iloc[:, :-2].astype(float))
            predictions = \
                df_autoencoders['autoencoder'][device].get_prediction(df_autoencoders['model'][device], test_feat2)[1]
            pred_mses = ((np.array(test_feat2) - np.array(predictions)) ** 2).mean(axis=1).reshape(-1, )
            final_mses[mac_add2] = pred_mses
            final_labels[mac_add2] = test_label2[0]
        tp_test2, fp_test2, tn_test2, fn_test2, precision2, tpr2, fpr2, f12, auc2, true_tmp2, pred_tmp2 = \
            compute_params_maj(final_mses, final_labels, device, th_new)
        df_results_majority.at[str(device) + "_1", 'th'] = th_new
        df_results_majority.at[str(device) + "_1", 'tst_TPR'] = tpr2
        df_results_majority.at[str(device) + "_1", 'tst_FPR'] = fpr2
        df_results_majority.at[str(device) + "_1", 'tst_Precision'] = precision2
        df_results_majority.at[str(device) + "_1", 'tst_F1'] = f12
        df_results_majority.at[str(device) + "_1", 'tst_ROC_AUC'] = auc2
        df_results_majority.at[str(device) + "_1", 'tst_Total_Session'] = test_label.shape[0]
        df_results_majority.at[str(device) + "_1", 'tp'] = tp_test2
        df_results_majority.at[str(device) + "_1", 'fp'] = fp_test2
        df_results_majority.at[str(device) + "_1", 'tn'] = tn_test2
        df_results_majority.at[str(device) + "_1", 'fn'] = fn_test2
        df_results_majority.at[str(device) + "_1", 'tst_1'] = 0
        df_results_majority.at[str(device) + "_0", 'tst_1'] = 0
        df_results_majority.at[str(device) + "_1", 'tst_0'] = 0
        df_results_majority.at[str(device) + "_0", 'tst_0'] = 0
        for idx in range(0, len(true_tmp2)):
            if true_tmp2[idx] == 1 and pred_tmp2[idx] == 1:
                df_results_majority.at[str(device) + "_1", 'tst_1'] += 1
            elif true_tmp2[idx] == 1 and pred_tmp2[idx] == 0:
                df_results_majority.at[str(device) + "_1", 'tst_0'] += 1
            elif true_tmp2[idx] == 0 and pred_tmp2[idx] == 0:
                df_results_majority.at[str(device) + "_0", 'tst_0'] += 1
            else:
                df_results_majority.at[str(device) + "_0", 'tst_1'] += 1
    df_results_majority.to_csv(os.path.join(out_path, "test_results_params_NORMAL363_maj.csv"))


def perf_measure(y_actual, y_pred):
    tp, fp, tn, fn = 0, 0, 0, 0
    for idx in range(len(y_pred)):
        if y_actual[idx] == y_pred[idx] == 1:
            tp += 1
        if y_pred[idx] == 1 and y_actual[idx] != y_pred[idx]:
            fp += 1
        if y_actual[idx] == y_pred[idx] == 0:
            tn += 1
        if y_pred[idx] == 0 and y_actual[idx] != y_pred[idx]:
            fn += 1
    return tp, fp, tn, fn


def compute_params_maj(final_mses, final_labels, label, th):
    tmp_pred_labels = []
    tmp_true_labels = []
    for key in final_mses.keys():
        lst = final_mses[key]
        for num in range(0, len(lst)):
            count = len([idx for idx in lst[num:num + 3] if idx <= th])
            if count >= 2:
                tmp_pred_labels.append(1)
            else:
                tmp_pred_labels.append(0)
            if final_labels[key] == label:
                tmp_true_labels.append(1)
            else:
                tmp_true_labels.append(0)
            if num+3 > len(lst):
                break
    tp, fp, tn, fn = perf_measure(tmp_true_labels, tmp_pred_labels)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = -1
    try:
        tpr = tp / (tp + fn)
    except ZeroDivisionError:
        tpr = -1
    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = -1
    try:
        f1 = 2 * (precision * tpr) / (precision + tpr)
    except ZeroDivisionError:
        f1 = -1
    fpr_test, tpr_test, thresholds_tst = metrics.roc_curve(y_true=tmp_true_labels, y_score=tmp_pred_labels, pos_label=1)
    auc = metrics.auc(x=fpr_test, y=tpr_test)
    return tp, fp, tn, fn, precision, tpr, fpr, f1, auc, tmp_true_labels, tmp_pred_labels


def compute_params(true_labels, pred_labels, label, th):
    tmp_true_labels = []
    tmp_pred_labels = []
    pred_labels = pred_labels.tolist()
    tl = true_labels.values.tolist()
    tl = [item[0] for item in tl]
    for num in range(0, len(tl)):
        if tl[num] == label:
            tmp_true_labels.append(1)
        else:
            tmp_true_labels.append(0)
        if pred_labels[num] <= th:
            tmp_pred_labels.append(1)
        else:
            tmp_pred_labels.append(0)
    tp, fp, tn, fn = perf_measure(tmp_true_labels, tmp_pred_labels)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = -1
    try:
        tpr = tp / (tp + fn)
    except ZeroDivisionError:
        tpr = -1
    try:
        fpr = fp / (fp + tn)
    except ZeroDivisionError:
        fpr = -1
    try:
        f1 = 2 * (precision * tpr) / (precision + tpr)
    except ZeroDivisionError:
        f1 = -1
    fpr_test, tpr_test, thresholds_tst, auc = -1, -1, -1, -1
    if 0 in tmp_true_labels:
        fpr_test, tpr_test, thresholds_tst = metrics.roc_curve(y_true=tmp_true_labels, y_score=pred_labels, pos_label=1)
        auc = metrics.auc(x=fpr_test, y=tpr_test)
    return tp, fp, tn, fn, precision, tpr, fpr, f1, auc, tmp_true_labels, tmp_pred_labels


def eval_test(test_feat22, test_labels):
    test_labels = test_labels.values.tolist()
    test_labels = [item[0] for item in test_labels]
    for row in df_autoencoders.itertuples():
        autoencoder_obj = df_autoencoders['autoencoder'][row.Index]
        predictions = autoencoder_obj.model.predict(test_feat22)
        mses = ((np.array(test_feat22) - np.array(predictions)) ** 2).mean(axis=1).reshape(-1, )
        th1 = df_autoencoders['th'][row.Index]
        for idx in range(len(mses)):
            if mses[idx] <= th1:
                df_confusion[test_labels[idx]][row.Index] += 1
    df_confusion.to_csv(os.path.join(out_path, "test_results_matrix_NORMAL363.csv"))
#  ------------------------------------------ END EVALUATION & RESULT PROCESS ---------------------------


#  --------------------------------------------------- PROJECT MAIN --------------------------------------
# pre_process_categories() // ENABLE FOR pre_process
feat_scale = MinMaxScaler()
# if os.path.isfile("best_model_camera.h5"):
#    model_exists = True
device_num = 1
for item_class in device_categories:
    print(f"{device_num} of {len(device_categories)} \n")
    device_num += 1
    train_feat, valid_feat, train_label, val_label = create_dictionaries(item_class)
    train_feat = train_feat.dropna(axis=1)
    valid_feat = valid_feat.dropna(axis=1)
    train_feat2 = feat_scale.fit_transform(train_feat.astype(float))
    valid_feat2 = feat_scale.fit_transform(valid_feat.astype(float))
    # th_hold = None
    if model_exists:
        df_autoencoders['autoencoder'][item_class] = autoencoder()
        df_autoencoders['model'][item_class] = load_model(f"best_model_{item_class}.h5")
        df_autoencoders['autoencoder'][item_class].get_saved_model(load_model(f"best_model_{item_class}.h5"))
        th_hold = df_autoencoders['autoencoder'][item_class].get_prediction(df_autoencoders['model'][item_class],
                                                                            valid_feat)[0]
        df_autoencoders['th'][item_class] = th_hold
    else:
        df_autoencoders['autoencoder'][item_class] = autoencoder()
        df_autoencoders['autoencoder'][item_class].create_model(train_feat)
        df_autoencoders['model'][item_class] = df_autoencoders['autoencoder'][item_class].model
        th_hold = df_autoencoders['autoencoder'][item_class].train_model(
            df_autoencoders['autoencoder'][item_class].model, train_feat2, valid_feat2)
        df_autoencoders['th'][item_class] = th_hold
        layer = df_autoencoders['autoencoder'][item_class].model.layers[3]
        weights = layer.get_weights()[0]
        print(layer.name)
        input(len(weights))
    # --------------------tr-----------------------------
    predictions_on_tr = \
        df_autoencoders['autoencoder'][item_class].get_prediction(df_autoencoders['model'][item_class], train_feat2)[1]
    predicted_mses = ((np.array(train_feat2) - np.array(predictions_on_tr)) ** 2).mean(axis=1).reshape(-1, )
    tp_tr, fp_tr, tn_tr, fn_tr, precision_tr, tpr_tr, fpr_tr, f1_tr, auc_tr, true_tmp_tr, pred_tmp_tr = compute_params(
        train_label, predicted_mses, item_class, th_hold)
    # --------------------val--------------------
    predictions_on_val = \
        df_autoencoders['autoencoder'][item_class].get_prediction(df_autoencoders['model'][item_class], valid_feat2)[1]
    predicted_mses = ((np.array(valid_feat2) - np.array(predictions_on_val)) ** 2).mean(axis=1).reshape(-1, )
    tp_val, fp_val, tn_val, fn_val, precision_val, tpr_val, fpr_val, f1_val, auc_val, true_tmp_val, \
    pred_tmp_val = compute_params(val_label, predicted_mses, item_class, th_hold)
    # ----------------------------------------------------------------------------

    df_results1.at[str(item_class) + "_1", 'train_TPR'] = tpr_tr
    df_results1.at[str(item_class) + "_1", 'train_Precision'] = precision_tr
    df_results1.at[str(item_class) + "_1", 'train_Total_Session'] = train_label.shape[0]
    df_results1.at[str(item_class) + "_1", 'tp_tr'] = tp_tr
    df_results1.at[str(item_class) + "_1", 'fp_tr'] = fp_tr
    df_results1.at[str(item_class) + "_1", 'tn_tr'] = tn_tr
    df_results1.at[str(item_class) + "_1", 'fn_tr'] = fn_tr
    df_results1.at[str(item_class) + "_1", 'val_TPR'] = tpr_val
    df_results1.at[str(item_class) + "_1", 'val_FPR'] = fpr_val
    df_results1.at[str(item_class) + "_1", 'val_Precision'] = precision_val
    df_results1.at[str(item_class) + "_1", 'val_Total_Session'] = val_label.shape[0]
    df_results1.at[str(item_class) + "_1", 'tp_val'] = tp_val
    df_results1.at[str(item_class) + "_1", 'fp_val'] = fp_val
    df_results1.at[str(item_class) + "_1", 'tn_val'] = tn_val
    df_results1.at[str(item_class) + "_1", 'fn_val'] = fn_val
    df_results1.at[str(item_class) + "_1", 'tst_1'] = 0
    df_results1.at[str(item_class) + "_1", 'tst_0'] = 0
    df_results1.at[str(item_class) + "_0", 'tst_1'] = 0
    df_results1.at[str(item_class) + "_0", 'tst_0'] = 0
    df_results1.at[str(item_class) + "_1", 'vld_1'] = 0
    df_results1.at[str(item_class) + "_1", 'vld_0'] = 0
    df_results1.at[str(item_class) + "_0", 'vld_1'] = 0
    df_results1.at[str(item_class) + "_0", 'vld_0'] = 0
    df_results1.at[str(item_class) + "_1", 'trn_1'] = 0
    df_results1.at[str(item_class) + "_1", 'trn_0'] = 0
    df_results1.at[str(item_class) + "_0", 'trn_1'] = 0
    df_results1.at[str(item_class) + "_0", 'trn_0'] = 0
    # __________________________train_______________________________________
    for i in range(0, len(true_tmp_tr)):
        if true_tmp_tr[i] == 1 and pred_tmp_tr[i] == 1:
            df_results1.at[str(item_class) + "_1", 'trn_1'] += 1
        elif true_tmp_tr[i] == 1 and pred_tmp_tr[i] == 0:
            df_results1.at[str(item_class) + "_1", 'trn_0'] += 1
        elif true_tmp_tr[i] == 0 and pred_tmp_tr[i] == 0:
            df_results1.at[str(item_class) + "_0", 'trn_0'] += 1
        else:
            df_results1.at[str(item_class) + "_0", 'trn_1'] += 1
    # _______________________________val __________________________________________
    for i in range(0, len(true_tmp_val)):
        if true_tmp_val[i] == 1 and pred_tmp_val[i] == 1:
            df_results1.at[str(item_class) + "_1", 'vld_1'] += 1
        elif true_tmp_val[i] == 1 and pred_tmp_val[i] == 0:
            df_results1.at[str(item_class) + "_1", 'vld_0'] += 1
        elif true_tmp_val[i] == 0 and pred_tmp_val[i] == 0:
            df_results1.at[str(item_class) + "_0", 'vld_0'] += 1
        else:
            df_results1.at[str(item_class) + "_0", 'vld_1'] += 1
    # __________________________________ test________________________________________
test_feat = pd.read_csv(os.path.join(out_path, "test_features.csv"))
test_feat, test_macs = separate_feat_label(test_feat)
test_feat_scaled = feat_scale.fit_transform(test_feat.astype(float))
test_label = pd.read_csv(os.path.join(out_path, "test_labels.csv"))
majority_test()
eval_test(test_feat_scaled, test_label)
for item_class in device_categories:
    predictions_on_tst = \
        df_autoencoders['autoencoder'][item_class].get_prediction(df_autoencoders['model'][item_class],
                                                                  test_feat_scaled)[1]
    predicted_mses = ((np.array(test_feat_scaled) - np.array(predictions_on_tst)) ** 2).mean(axis=1).reshape(-1, )
    th_hold = df_autoencoders['th'][item_class]
    tp_test, fp_test, tn_test, fn_test, precision_test, tpr_tst, fpr_tst, f1_tst, auc_tst, true_tmp_tst, pred_tmp_tst =\
        compute_params(test_label, predicted_mses, item_class, th_hold)
    df_results1.at[str(item_class) + "_1", 'tst_TPR'] = tpr_tst
    df_results1.at[str(item_class) + "_1", 'tst_FPR'] = fpr_tst
    df_results1.at[str(item_class) + "_1", 'tst_Precision'] = precision_test
    df_results1.at[str(item_class) + "_1", 'tst_F1'] = f1_tst
    df_results1.at[str(item_class) + "_1", 'tst_ROC_AUC'] = auc_tst
    df_results1.at[str(item_class) + "_1", 'tst_Total_Session'] = test_label.shape[0]
    df_results1.at[str(item_class) + "_1", 'tp'] = tp_test
    df_results1.at[str(item_class) + "_1", 'fp'] = fp_test
    df_results1.at[str(item_class) + "_1", 'tn'] = tn_test
    df_results1.at[str(item_class) + "_1", 'fn'] = fn_test
    for i in range(0, len(true_tmp_tst)):
        if true_tmp_tst[i] == 1 and pred_tmp_tst[i] == 1:
            df_results1.at[str(item_class) + "_1", 'tst_1'] += 1
        elif true_tmp_tst[i] == 1 and pred_tmp_tst[i] == 0:
            df_results1.at[str(item_class) + "_1", 'tst_0'] += 1
        elif true_tmp_tst[i] == 0 and pred_tmp_tst[i] == 0:
            df_results1.at[str(item_class) + "_0", 'tst_0'] += 1
        else:
            df_results1.at[str(item_class) + "_0", 'tst_1'] += 1
df_results1.to_csv(os.path.join(out_path, "test_results_params_NORMAL363.csv"))
print("$$$$$$$$$$$$$$ DONE SUCCESSFULY $$$$$$$$$$$$$$")
# TODO: talk with eliya about running anaconda from CMD (activate env) -> emailed yair for permittion
# TODO: check for script to run system on various hyper parameters
# TODO: check if early stop + best model is better
# TODO: yair email
