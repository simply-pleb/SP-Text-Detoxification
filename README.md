# SP-Text-Detoxification

- author: Ahmadsho Akdodshoev
- email: a.akdodshoev@innopolis.university
- group #: BS20AAI

## Usability

This solution works on python version 3.9.18. All the requirements are stated in the requirements file.

```
pip install -r requirements.txt
```

In order to transform the data use 

```
python3 src/data/make_dataset.py
```

this command will select data from paranmt dataset and merge that with paradetox dataset.

To train the model use

```
python3 src/models/train_model.py
```

this will create and save a dataset dictionary with train, test and eval data, and then train the model. Model specifics:

- BART-base for conditional generation
- dataset of 20,000 texic/neutral pairs. About 12k from paradetox and 8k from paranmt datasets respectively.
- 20 epochs
- Adam optimizer
- learning rate of -3e5

Predict using the model  
```
python3 src/models/predict_model.py
```
will load the test data, generate sequences for it and save both inputs and predictions to specified files for evaluating the model performance using metric.py (in src/metric).

To predict your own sequence use 
```
python3 src/models/predict_model.py --pred "you-sequence..."
```
or use explore-data notebook. it should be faster to use.

### References

- [Text Detoxification using Large Pre-trained Neural Models (2021)](https://aclanthology.org/2021.emnlp-main.629/)
- [ParaDetox: Detoxification with Parallel Data (2022)](https://aclanthology.org/2022.acl-long.469/)
- [Text Style Transfer with Contrastive Transfer Pattern Mining (2023)](https://aclanthology.org/2023.acl-long.439/)
- [DiffuDetox: A Mixed Diffusion Model for Text Detoxification (2023)](https://arxiv.org/abs/2306.08505)
<!-- - [Deep Learning for Text Style Transfer: A Survey (2022)](https://direct.mit.edu/coli/article/48/1/155/108845/Deep-Learning-for-Text-Style-Transfer-A-Survey) -->