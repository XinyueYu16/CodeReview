import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier


class SimCityPredictor():
  '''
  人口元宇宙相关模型，包含清洗+筛选入模数据，建模，对可预测数据进行预测，数据整理
  - 模型可替换
  - 数据输入要求发生改变时，只需继承父类覆盖def
  '''
  def __init__(self):
    self.label = 'hedu'
    self.feature_col = ['hage', 'hgender']
    self.cate_col = ['hedu', 'hgender']
    self.output_path = "../python_test_output_0711/"
    self.input_name = "sim_shanghai_clean.dta"
    self.model_name = f"_model_{self.label}.pkl"
    self.n_pad = 10
    self.clf = DecisionTreeClassifier(
      max_depth=10,
      min_samples_leaf=5,
      random_state=42)

  def preprocess(self, df):
      return df.copy()

  def select_inmodel(self):
    pass

  def return_metric(true, pred):
      return (f1_score(true, pred, average='macro'),
              recall_score(true, pred, average='macro'),
              precision_score(true, pred, average='macro'),
              accuracy_score(true, pred))

  def filter_hasres(self, df):
    return df.dropna(subset=self.feature_col).index

  def predict_res(self):
    pass

  def process_n_predict(self, df, to_return_metric=True):
    data = self.preprocess(df)
    inmodel = self.select_inmodel(data)
    Xy = inmodel.copy()
    for col in self.cate_col:
      Xy[col] = pd.Categorical(Xy[col], ordered=True)

    self.clf.fit(Xy[self.feature_col], Xy[f'{self.label}'])
    # 输出模型效果
    if to_return_metric:
      metric = {}
      metric['input_size'] = Xy.shape[0]
      metric['log_loss'] = log_loss(Xy[f'{self.label}'].values,
                                    self.clf.predict_proba(Xy[self.feature_col]))
      metric['f1_score'], metric['recall'], metric['precision'], metric[
          'accuracy'] = self.return_metric(Xy[f'{self.label}'],
                                      self.clf.predict(Xy[self.feature_col]))
      print(metric)

    # 保存pkl
    with open(os.path.join(self.output_path,
                           f'{self.input_name.replace(".dta", self.model_name)}'
                           ), 'wb') as f:
      pickle.dump(self.clf, f)

    # 预测可预测部分，保留剩余不动
    residx = data.pipe(self.filter_hasres)
    r_hres = data.loc[residx].copy()
    r_nores = data.loc[data.index.difference(residx)].copy()
    if len(residx) > 0:
      tempres = (
          pd.concat(
              [r_hres.reset_index(drop=True),
              pd.DataFrame(self.clf.predict_proba(r_hres[self.feature_col]),
                          columns=Xy[f'{self.label}'].cat.categories)
              ],axis=1))
      tempres.columns = [str(i).rjust(self.n_pad, "0")
                          if type(i) == float else i
                          for i in tempres.columns]
      #在各level上的累计概率
      for idx in range(len(Xy[f'{self.label}'].cat.categories)):
        current_level = (str(Xy[f"{self.label}"].cat.categories[idx])
                          .rjust(self.n_pad, "0"))
        tempres[f'acc_{current_level}'] = tempres[
          [str(Xy[f"{self.label}"].cat.categories[idx_]).rjust(self.n_pad, "0")
          for idx_ in range(len(list(Xy[f'{self.label}'].cat.categories)))
          if idx_ <= idx]].sum(axis=1)

      np.random.seed(42)
      tempres['uniform'] = np.random.uniform(0, 1, len(tempres))
      pyres_ = tempres.reset_index()
      long = (
          pyres_
          .melt(id_vars=['index', 'uniform'], var_name='acc',
              value_vars=[i for i in tempres.columns if i.find('acc') != -1])
          .query('uniform <= value')
          .sort_values(['index', 'acc'])
          .groupby('index')
          .head(1))

      pyres = (
          pyres_
          .merge(long[['index', 'acc']]
                  .rename({'acc': f'{self.label}hat_'}, axis=1),
                on='index')
          .assign(**{f'{self.label}hat': lambda df:
                  (df[f'{self.label}hat_']
                    .str.replace('acc_','').astype(float))})
          .pipe(self.predict_res)
          .pipe(lambda df:
              df.drop([i for i in df.columns
                      if type(i) == float or i.find('.') != -1], axis=1))
          .append(r_nores)
          .drop(['index', 'uniform',
                 f'{self.label}hat_', f'{self.label}hat'], axis=1)
          )

      for col in self.cate_col:
          pyres[col] = pyres[col].astype(float)
    else:
      pyres = data.copy()
    return pyres.reset_index(drop=True)


class MallFootfallPredictor():
  pass
