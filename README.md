# 训练
```
python bin/train_v2.py config/example.json /root/runs1/ --device_ids 0
```
训练测试和验证用的数据集存放在images路径下

# 测试
```
python bin/test.py /root/runs/ train.csv train_out.csv
```

# ROC曲线
```
python bin/roc.py train_out.csv train.csv . Banana
```