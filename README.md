# Finding the RLN: Using Deep Learning to Identify the Recurrent Laryngeal Nerve During Thyroidectomy
This is the official code repository for "[Using deep learning to identify the recurrent laryngeal nerve during thyroidectomy](https://www.nature.com/articles/s41598-021-93202-y)", _Scientific Reports_ 2021. You can also find our project page [here](http://web.stanford.edu/~jxgong/rln.html), and a Medium blog on our work [here](https://medium.com/hasty-ai/using-computer-vision-to-identify-critical-anatomy-in-surgical-images-b1cea20f365c).

## Requirements
The packages required and their versions are listed in `requirements.txt`.

You can create a compatible conda environment using the command:

`conda create --name finding-rln --file requirements.txt -c default -c pytorch python=3.6`.

Note that this code is not tested using other versions of these packages. This code was only tested on a TITAN RTX GPU with 24G memory.

## File structure
The code is located in `src`, while the data files and dataset should be placed in `data`.
```
data
├── annotations
├── images
├── image_crops.pkl
├── ...
├── retractor_contours.pkl
src
├── utils
│   ├── dataset_crop.py
│   ├── ...
│   ├── dataset_retractor.py
├── dice_loss.py
├── ...
├── train_retractor.py
```

## Dataset
Place the dataset masks (annotations) in `data/annotations` and the images in `data/images` as noted above. Leave the remaining `.pkl` files as-is.

Note that the index in the .png masks for the nerve class is 1, and for the retractor class, we combine indices 15, 19, and 20. The background index is 0. All other indices indicate other classes that are not used in this paper.

## Training and testing
All code files are named after the convention `<phase>_<task>.py`. For instance, to train a nerve segmentation model, you would use `train_nerve.py`.

### Training
To **train** a model, use:

`python3 train_<task>.py -e <num_epochs> -b <batch_size> -l <initial_learning_rate> -s 0.25 -d <checkpoint_dir> -t <test_fold>`

The flag `-s` denotes the image scale; in our method, we use 0.25. For `-t`, use the fold of the dataset (from 1 to 5 inclusive) that this model would be tested on at inference.

Note that the code currently trains on all 4 train/val splits (as opposed to holding out val for validation). You can adjust this by modifying the dataset file in `src/utils` (e.g. line 83 in `dataset_nerve.py`, etc.) and `get_sampler` in the training script (e.g. line 43 in `train_nerve.py`).

### Testing
To **test** a model, use:

`python3 test_<task>.py -s 0.25 -m <path_to_checkpoint.pth> -t <test_fold>`.

## Citation
If you use our code, please consider citing our paper:
```
@article{gong2021rln,
  title = {Using deep learning to identify the recurrent laryngeal nerve during thyroidectomy},
  author = {Gong, Julia and Holsinger, Christopher F. and Noel, Julia E. and Mitani, Sohei and Jopling, Jeff and Bedi, Nikita and Koh, Yoon Woo and Orloff, Lisa A. and Cernea, Claudio R. and Yeung, Serena},
  journal = {Scientific Reports},
  volume = {11},
  number = {14306},
  year = {2021},
  doi = {10.1038/s41598-021-93202-y}
}
```

## Contact
Please contact the co-corresponding authors, F. Christopher Holsinger and Serena Yeung, for the dataset and other inquiries as noted in the paper link.

## Acknowledgements
This repository uses modified code from [this pytorch repository](https://github.com/milesial/Pytorch-UNet).
