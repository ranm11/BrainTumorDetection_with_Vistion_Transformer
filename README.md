Vision transformer was initially published in 2020.
In this workshop wwe'll try to classify brain tumors based on the Kaggle dataset 
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset.

Runnig out model on Dataset we face overfit since ~250 MRI scans images are too shallow for the transformer model.
following Data Augmentation, we enlarge our dataset to ~500 images and reach  more than 90% accuracy in test cases.

as first step we devide our image in to patches 

project each patch via 
