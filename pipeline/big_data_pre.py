import pandas as pd
import numpy as np
from datetime import datetime

# data = pd.read_csv('', header=None)  # ����ͨ��header=None�Լ�ָ������
#
# data.columns = ['label', 'uId', 'adId', 'operTime', 'siteId', 'slotId', 'contentId', 'netType']

# ����ָ�����͡�ת��ʱ�䣬���²������ڴ�ռ�ý��͵���13.4GB
data = pd.read_csv('train_20190518.csv',
                   header=None,
                   names=['label', 'uId', 'adId', 'operTime', 'siteId', 'slotId', 'contentId', 'netType'],
                   dtype={'label': np.bool, 'uId': np.object, 'adId': np.uint32, 'operTime': np.object, 'siteId': np.uint8, 'slotId': np.uint32, 'contentId':np.uint32, 'netType': np.uint8},
                   parse_dates=['operTime'])

data['adId'].argmax()

# �鿴ÿ�е����������Լ���������Ҫ���ڴ�ռ�
data.info(memory_usage='deep')


# def mem_usage(pandas_obj):
#     if isinstance(pandas_obj, pd.DataFrame):
#         usage_b = pandas_obj.memory_usage(deep=True).sum()
#     else:  # ���Ǽ����ⲻ��һ��df������һ�� Series
#         usage_b = pandas_obj.memory_usage(deep=True)
#     usage_mb = usage_b / 1024 ** 2  # �� bytes ת���� megabytes
#     return "{:03.2f} MB".format(usage_mb)
#
# #  np.iinfo(it) ָ�����͵ķ�Χ��itָ"uint8"֮����ַ���
#
#
# data_int = data.select_dtypes(include=['int'])  # �� DataFrame.select_dtypes ��ѡ�б��е� int����
# converted_int = data_int.apply(pd.to_numeric, downcast='unsigned')  # ��pd.to_numeric()���������ǵ���������
#
# print(mem_usage(data_int))
# print(mem_usage(converted_int))
#
# compare_ints = pd.concat([data_int.dtypes, converted_int.dtypes], axis=1)
# compare_ints.columns = ['before', 'after']
# compare_ints.apply(pd.Series.value_counts)

# ���ڶ�ȡ�����ݣ��õ�����������Ӧ��������
from datetime import datetime
times = pd.DataFrame([datetime.strftime(x, "%Y-%m-%d") for x in data['operTime']])
times.value_counts()  # ͳ�Ƹ������ڳ��ִ���

# df.resample('w').sum().head()����Ҳ����

# ����������ʾ��ֻѡ��2019-03-26��2019-03-31������
# 2019-03-30    28043936
# 2019-03-31    28028243
# 2019-03-29    27794906
# 2019-03-26    25583297
# 2019-03-27    25300266
# 2019-03-28    24807386
# 2019-03-25      138692
# 2019-03-24       52724
# 2019-03-23       23893
# 2019-03-22       16417
# 2019-03-21       11739
# 2019-03-20        8683
# 2019-03-19        5965
# 2019-03-18        3623
# 2019-04-01        2854
# 2019-03-17        2837
# 2019-03-16        2401
# 2019-03-15        1757
# 2019-03-14        1382
# 2019-03-13        1029
# 2019-03-12         897
# 2019-03-11         797
# 2019-03-10         691
# 2019-03-09         521
# 2019-03-08         517
# 2019-03-07         413
# 2019-03-06         244
# 2019-03-03         161
# 2019-03-02         155
# 2019-03-05         145

# ɸѡ�ض�ʱ��εģ����Ҷ�����
