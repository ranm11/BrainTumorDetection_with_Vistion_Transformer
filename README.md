Vision transformer was initially published in 2020.
In this workshop wwe'll try to classify brain tumors based on the Kaggle dataset 
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset.

Runnig out model on Dataset we face overfit since ~250 MRI scans images are too shallow for the transformer model.
following Data Augmentation, we enlarge our dataset to ~500 images and reach  more than 90% accuracy in test cases.

as first step we devide our image in to patches 

![patch_sequence_yes_no_brain](https://github.com/user-attachments/assets/c7e2348d-3d99-470e-af61-ed4f1f9b57ab)

then we project each patch via input layer (Embedding layer) , and each patch shall get a vector of 256 (dmodule) len.
following embedding layer each patch shall go through Positional Embedding to mak up sequence.

once we have embeded values  within positional encoding we can apply K Q V values to multi head attention layer,
and get attention map.
from attention map we may extract the heat map :

![image](https://github.com/user-attachments/assets/22801f65-50e3-4877-a3c2-14b9069e6c04)

![positional_encoding](https://github.com/user-attachments/assets/62305fc8-7516-4a54-8581-aa1135d25ff4)


