This model use many dataset to train and finetuned. 

## Cut dataset 

dataset that contain 22 endangered animals in Viet Nam(approximately 40 images each). [animal cut dataset](Config/data1.yaml)
```
!gdown --id 1OpMdJdcvhDIuR3H98OpeNWEo8IyryE_k -O dataset.zip
!unzip -q dataset.zip -d /content/dataset/
```
or download direct here [animal cut dataset](https://drive.google.com/file/d/1OpMdJdcvhDIuR3H98OpeNWEo8IyryE_k/view?usp=drive_link)

## Generated dataset for finetuned model

dataset that genereated my GAN model from the animal cut dataset, approximately 1.1k images. 
[gen data](https://drive.google.com/file/d/12L6gyaWAAd8yT33aTs_vNv4OuylxKkn5/view?usp=drive_link)

## Fine-tune yolo11 dataset

 dataset that contain 22 endangered animals in Vietnam. original iamges uncut version, animal with background, got preprocessed manually on roboflow with bounding box and annotated "animal" each image. 
 
 [animal uncut dataset for finetune yolo](https://drive.google.com/drive/folders/1D7fL16ZK1o86agnTVCFmfdHjAH54Xze_?usp=drive_link)

 And also use gen dataset previous with same data preprocessing. 

[config data](Config/datagen.ymal.) 

## Fine-tune Mobilevnet3

[dataset for mb3 and clip](https://drive.google.com/file/d/1Tn08BPAPoCVI_kC3N-9x-Z9a2kULa6eQ/view?usp=sharing)
