

def run_models(file, model, to_use, enc="onehot"):
    
    import copy
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.random import uniform
    import pandas as pd
    import xgboost as xgb
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import roc_curve, mean_absolute_error, plot_roc_curve, auc
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate, KFold
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import DMatrix, cv
    import scipy.sparse as sp
    
    
    def train_model(X, y, params, model, n_jobs=-1, cv=3, n_params=5, ret_score=False):
        crossv_mod = copy.deepcopy(model)
        ret_mod = copy.deepcopy(model)
    
        grid = RandomizedSearchCV(model, params, cv=cv, scoring='neg_median_absolute_error', verbose=0, n_jobs=n_jobs,
                                  n_iter=n_params, refit=False)
        grid.fit(X, y)
    
        cv_pred = KFold(n_splits=cv)
    
        # Use the same parameters for the training set to get CV predictions
        crossv_mod.set_params(**grid.best_params_)
        s = cross_validate(crossv_mod, X=X, y=y, cv=cv_pred, n_jobs=n_jobs, verbose=0, return_train_score=True)
        scores = [np.mean(s["train_score"]), np.mean(s["test_score"])]
    
        # Train the final model
        ret_mod.set_params(**grid.best_params_)
        ret_mod.fit(X, y)
    
        if ret_score:
            return scores
    
        return ret_mod
    
    # ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~
    
    class XGBoostModel:
        def __init__(self, regressor=False):
            self.parameters = {'n_estimators': list(range(10, 200, 1)), 'max_depth': list(range(1, 12)),
                               'learning_rate': list(uniform(0.01, 0.25, 100)), 'gamma': list(uniform(0, 10, 100)),
                               'reg_alpha': list(uniform(0, 10, 100)), 'reg_lambda': list(uniform(0, 10, 100)),
                               'scale_pos_weight': list(range(1, 100, 1))
                               }
            self.model = xgb.XGBClassifier(**self.parameters)
            if regressor:
                self.model = xgb.XGBRegressor(**self.parameters)
            self.model_cv = None
    
        def fit(self, data=None, values=None, dm=None, metric="median_absolute_error"):
            if dm is None:
                dm = DMatrix(data, label=values)
                self.model.fit(data, values)
                self.model_cv = cv(params=self.parameters, dtrain=dm)
    
        def get_metrics(self):
            if self.model_cv is None:
                raise Exception("Model should be computed first")
            return self.model_cv.tail(1)
    
        def get_importance(self):
            return self.model.feature_importances_
    
    class NeuralNet:
        def __init__(self):
            self.parameters = {"activation": ["relu", "sigmoid"]}
    
            def create_model(in_shape):
                model = models.Sequential()
                model.add(Dense(32, kernel_regularizer=regularizers.l2(0.003), activation='relu', input_shape=(in_shape,)))
                model.add(Dropout(0.5))
                model.add(Dense(16, kernel_regularizer=regularizers.l2(0.003), activation='relu'))
                model.add(Dropout(0.6))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
                return model(**self.parameters)
    
            self.model = create_model()
    
        def fit(self, data=None, values=None):
            x_partial_train, y_partial_train, x_validation, y_validation = train_test_split(data, values)
            self.model.fit(x_partial_train, y_partial_train, validation_data=(x_validation, y_validation))
    
        def predict(self, test_data):
            probs = self.model.predict_proba(test_data)
            return probs
    
    class SVMModel:
        def __init__(self):
            self.parameters = {'C': list(uniform(0.01, 300.0, 1000)), 'dual': [True],
                               'fit_intercept': [True, False], 'intercept_scaling': list(uniform(0, 1.0, 100)),
                               'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'], 'max_iter': [100000],
                               'tol': [0.001], 'verbose': [0]}
    
            self.model = LinearSVC(**self.parameters)
            self.model_cv = None
    
        def fit(self, data=None, values=None, dm=None, metric="median_absolute_error"):
            if dm is None:
                dm = DMatrix(data, label=values)
                self.model.fit(data, values)
                self.model_cv = cv(params=self.parameters, dtrain=dm)
    
        def get_metrics(self):
            if self.model_cv is None:
                raise Exception("Model should be computed first")
            return self.model_cv.tail(1)
    
        def get_importance(self):
            return self.model.coef_
    
    def plot_roc(model, X, y, folds=5):
    
        cv = StratifiedKFold(n_splits=folds)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
    
        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(cv.split(X, y)):
            viz = plot_roc_curve(model, X.iloc[test], y.iloc[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
    
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.legend(loc="lower right")
        plt.savefig("AUC.png", dpi=120)
    
    # ~~~~~~~ Reading data ~~~~~~~~~~~~~~~~~~
    features = file
    target = features.pop("label")
    if model == "XGB": model = XGBoostModel()
    if model == "SVM": model = SVMModel()
    
    # ~~~~~~~ Encodings ~~~~~~~~~~~~~~~~~~
    
    # One Hot Encoding
    if enc == "onehot":
        to_encode = features[[to_use]]
        temp_data = [pd.get_dummies(to_encode[col].apply(lambda x: pd.Series(list(x))), prefix=col) for col in to_encode.columns]
        features_oneHot = pd.concat(temp_data, sort=False, axis=1)
        # temporary solution
        features_oneHot.columns = [i for i in range(features_oneHot.shape[1])]
        features = features_oneHot
        
    # K-mer encoding
    if enc == "kmer":
        def getKmers(sequence, size=3): return np.array([sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)])
        features_kmer = features[[to_use]].select_dtypes(exclude=["int", "float"]).applymap(lambda x: " ".join(getKmers(x)))
        cv = CountVectorizer()
        features_kmer = pd.DataFrame(sp.hstack(features_kmer.apply(lambda col: cv.fit_transform(col))).toarray())
        features = features_kmer

    
    if model != "NN":
        # ~~~~~~~ Train test split ~~~~~~~~~~~~~~~~~~
        trainX, testX, trainy, testy = train_test_split(features, target)

        # ~~~~~~~ Training ~~~~~~~~~~~~~~~~~~
        best_model = train_model(trainX, trainy, model.parameters, model.model)

        # ~~~~~~~ AUC ROC curve ~~~~~~~~~~~~~~~~~~
        plot_roc(best_model, testX, testy)

        if model == "XGB":
            # ~~~~~~~ feature importance ~~~~~~~~~~~~~~~~~~
            plt.plot(best_model.feature_importances)
            plt.savefig("feat_imp", dpi=120)


# cv = StratifiedKFold(n_splits=5)
# X = features_oneHot
# y = target
#
# def create_model():
#     input_seq = keras.Input(shape=(None,), dtype="int32")
#     query_input = input_seq
#     value_input = input_seq
#     token_embedding = Embedding(12, 1, input_length=features_oneHot.shape[1])  # of shape [batch_size, Tq, dimension]
#     query_embeddings = token_embedding(query_input)  # Value embeddings of shape [batch_size, Tv, dimension]
#     value_embeddings = token_embedding(value_input)
#
#     cnn_layer = Conv1D(filters=100, kernel_size=4, padding='same')
#     query_seq_encoding = cnn_layer(query_embeddings)  # of shape [batch_size, Tq, filters]
#     value_seq_encoding = cnn_layer(value_embeddings)  # of shape [batch_size, Tv, filters]
#     query_value_attention_seq = Attention()([query_seq_encoding, value_seq_encoding])
#
#     # Reduce over the sequence axis to produce encodings of shape [batch_size, filters]
#     query_encoding = GlobalAveragePooling1D()(query_seq_encoding)
#     query_value_attention = GlobalAveragePooling1D()(query_value_attention_seq)
#
#     # Concatenate query and document encodings to produce a DNN input layer.
#     input_layer = Concatenate()([query_encoding, query_value_attention])
#
#     x = Dense(128, activation="relu")(input_layer)
#     preds = Dense(2, activation="softmax")(x)
#     model = keras.Model(input_seq, preds)
#     model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
#     model.summary()
#
#     return model
#
# model_clf = KerasClassifier(build_fn=create_model)
# model_clf.fit(X, y)
# model_clf.predict_proba(testX)
#
# fpr2, tpr2, threshold = roc_curve(testy, model_clf.predict_proba(testX)[:, 1])
# roc_auc2 = auc(fpr2, tpr2)
#
# def plot_for_nn(model, X, y):
#     aucs = []
#     for i, (train, test) in enumerate(cv.split(X, y)):
#         model.fit(X.iloc[train], y.iloc[train], batch_size=300, epochs=1, verbose=2)
#         probs = model.predict(X.iloc[test])[:, 1]
#         fpr, tpr, thresholds = roc_curve(y.iloc[test], probs)
#         aucs.append(auc(fpr, tpr))
#         plt.plot(fpr, tpr)
#     roc_auc = np.mean(aucs)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(["mean AUC {0:.{1}f}".format(roc_auc, 3)])
#     plt.show()
#
# plot_for_nn(model_clf, X, y)

    if model == "NN":
        import keras
        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.layers import Dense, Dropout, Activation, Conv1D, Attention, Input, Embedding, GlobalAveragePooling1D, Concatenate, MaxPooling1D, GlobalMaxPooling1D
        from keras import models, layers
        from keras import regularizers, losses, optimizers
        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.metrics import auc

        # define the model
        def build_model():
            model = Sequential()
            model.add(Dense(120, input_dim=features.shape[1], activation='relu'))
            model.add(Dense(40, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        keras_model = build_model()

        # ~~~~~~~ Train test split ~~~~~~~~~~~~~~~~~~
        trainX, testX, trainy, testy = train_test_split(features, target)

        keras_model.fit(trainX, trainy, epochs=20, batch_size=100, verbose=1)

        fpr_keras, tpr_keras, threshold = roc_curve(testy, keras_model.predict(testX))

        auc_keras = auc(fpr_keras, tpr_keras)

        plt.figure(1,figsize=(100, 100))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig("AUC.png", dpi=120)
