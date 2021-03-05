
def run_models(file, model_name, to_use, enc="onehot"):
    
    import copy
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.random import uniform
    import pandas as pd
    import xgboost as xgb
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import plot_roc_curve
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate, KFold
    from sklearn.svm import LinearSVC
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.metrics import auc
    from xgboost import DMatrix, cv
    import scipy.sparse as sp
    
    
    def train_model(X, y, params, model, n_jobs=-1, cv=3, n_params=5, ret_score=False):
        if not isinstance(model, Sequential):
            crossv_mod = copy.deepcopy(model)
            ret_mod = copy.deepcopy(model)

            grid = RandomizedSearchCV(model, params, cv=cv,
                                      scoring='neg_median_absolute_error', verbose=0, n_jobs=n_jobs,
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

            return ret_mod

        else:
            model.fit(X, y)
            return model


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
        def __init__(self, dimension=None):
            self.parameters = {"activation": ["relu", "sigmoid"]}

            def build_model():
                model = Sequential()
                model.add(Dense(120, input_dim=dimension, activation='relu'))
                model.add(Dense(40, activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                return model

            self.model = build_model()

        def fit(self, data=None, values=None, epochs=20, batch_size=100):
            trainX, testX, trainy, testy = train_test_split(data, values)
            self.model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)

        def predict(self, test_data):
            probs = self.model.predict(test_data)
            return probs
    
    class SVMModel:
        def __init__(self):
            self.parameters = {'C': list(uniform(0.01, 300.0, 1000)), 'dual': [True],
                               'fit_intercept': [True, False], 'intercept_scaling': list(uniform(0, 1.0, 100)),
                               'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'], 'max_iter': [100000],
                               'tol': [0.001], 'verbose': [0]}
    
            self.model = LinearSVC(**self.parameters)
            self.model_cv = None
    
        def fit(self, data=None, values=None, dm=None):
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
        plt.savefig("AUC.png", dpi=110)
    
    # ~~~~~~~ Reading data ~~~~~~~~~~~~~~~~~~
    features = file
    target = features.pop("label")

    # ~~~~~~~ Encodings ~~~~~~~~~~~~~~~~~~
    
    # One Hot Encoding
    if enc == "onehot":
        to_encode = features[[to_use]]
        temp_data = [pd.get_dummies(to_encode[col].apply(lambda x: pd.Series(list(x)))) for col in to_encode.columns]
        features_onehot = pd.concat(temp_data, sort=False, axis=1)
        # temporary solution
        # features_onehot.columns = [i for i in range(features_onehot.shape[1])]
        encoded = features_onehot
        
    # K-mer encoding
    if enc == "kmer":
        def getKmers(sequence, size=3): return np.array([sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)])
        features_kmer = features[[to_use]].select_dtypes(exclude=["int", "float"]).applymap(lambda x: " ".join(getKmers(x)))
        cv = CountVectorizer()
        features_kmer = pd.DataFrame(sp.hstack(features_kmer.apply(lambda col: cv.fit_transform(col))).toarray())
        encoded = features_kmer

    # ~~~~~~~ Model creation ~~~~~~~~~~~~~~~~~~
    if model_name == "XGB": model = XGBoostModel()
    if model_name == "SVM": model = SVMModel()
    if model_name == "NN": model = NeuralNet(dimension=encoded.shape[1])

    # ~~~~~~~ Train test split ~~~~~~~~~~~~~~~~~~
    trainX, testX, trainy, testy = train_test_split(encoded, target)

    # ~~~~~~~ Training ~~~~~~~~~~~~~~~~~~
    best_model = train_model(trainX, trainy, model.parameters, model.model)

    # ~~~~~~~ AUC ROC curve ~~~~~~~~~~~~~~~~~~
    if model_name != "NN":
        plot_roc(best_model, testX, testy)

    if model_name == "XGB":
        # ~~~~~~~ feature importance ~~~~~~~~~~~~~~~~~~

        len_unique = [sum(encoded.columns.str.startswith((str(i) + "_"))) for i in range(encoded.shape[1])]

        d = {'name': encoded.columns,
             'value': best_model.feature_importances_}
        imp = pd.DataFrame(data=d)

        imp.to_csv("data/feat_imp.csv", index_label=False)

        plt.plot(best_model.feature_importances_)
        plt.savefig("feat_imp", dpi=110)

    trainX, testX, trainy, testy = train_test_split(encoded, target)


    # disp = plot_precision_recall_curve(best_model, testX, testy)
    # y_score = best_model.decision_function(testX)
    #
    # from sklearn.metrics import average_precision_score
    # average_precision = average_precision_score(y_test, y_score)
    #
    # disp.ax_.set_title('2-class Precision-Recall curve: '
    #                    'AP={0:0.2f}'.format(average_precision))

    return best_model.predict(testX).flatten(), testy.values
