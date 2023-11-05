To evaluate your predictions run: 

```python metric.py --inputs PATH_TO_INPUTS --preds PATH_TO_PREDS```

Both inputs and predictions should be plain text files with one comment per line.

[source: https://github.com/s-nlp/detox/tree/main/emnlp2021/metric](https://github.com/s-nlp/detox/tree/main/emnlp2021/metric)

prerequisites:

- Wieting subword-embedding SIM model can be found [here](https://storage.yandexcloud.net/nlp/wieting_similarity_data.zip)
    - there is a method in metric.py that downloads it
- CoLA classifier can be found [here](https://drive.google.com/drive/folders/p6_3lCbw3J0MhlidvKkRbG73qwmtWuRp)

