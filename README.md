# Al-Si Alloy ER Extract Dataset and Code
There are two parts in the Al-Si Alloy ER Extract Dataset and Code, our dataset is in the Al-Si ER Extract Dataset folder and the code is in the code folder.
## Operating environment
CUDA	10.2	numpy	1.19.4
CuDNN	7.6.5	sklearn	0.22
Python	3.6.12	prettytalbe	0.7.0
Tensorflow	1.15.0	pandas	0.24.2
gensim	3.4.0		

## Al-Si Alloy ER Extraction Dataset
Al-Si alloy entity–relation extraction dataset, the dataset are randomly divided into a training set (train.txt) and a test set (test.txt) in a ratio of 8:1, 
and one-eighth of the sentences in the training set are randomly selected as the validation set (dev.txt).

## Code
### ELMo training
The file run.txt contains the training launcher, test launcher, weights file storage launcher, which can be started by copying it to the run.sh file.
### Joint extraction model
Run the program: run.sh

Configuration file: bio_config.txt