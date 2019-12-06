# Reference
- <a href="https://hoya012.github.io/blog/Fast-Style-Transfer-Tutorial/">  </a>
- <a href="https://github.com/pytorch/examples/tree/master/fast_neural_style" target="_blank"> pytorch example code </a>
- <a href="https://ezgif.com/video-to-gif" target="_blank"> avi --> gif for demo </a>
- <a href="https://colab.research.google.com/" target="_blank"> google colaboratory </a>

# fast-style-transfer-tutorial-pytorch
Simple Tutorials &amp; Code Implementation of fast-style-transfer(Perceptual Losses for Real-Time Style Transfer and Super-Resolution, 2016 ECCV) using PyTorch. This code is based on [pytorch example codes](https://github.com/pytorch/examples/tree/master/fast_neural_style)

### Style Image 
<p align="center">
  <img width="700" src="https://github.com/carpediem804/fast_style_transfer_pythorch/blob/master/style_image.jpg">
</p>

### Style Transfer Result image (original / output)
<p align="center">
  <img width="700" src="https://github.com/carpediem804/fast_style_transfer_pythorch/blob/master/test_img.jpg">
</p>

<p align="center">
  <img width="700" src="https://github.com/carpediem804/fast_style_transfer_pythorch/blob/master/changed2_img.png">
</p>

<p align="center">
  <img width="700" src="https://github.com/carpediem804/fast_style_transfer_pythorch/blob/master/test5.jpg">
</p>

<p align="center">
  <img width="700" src="https://github.com/carpediem804/fast_style_transfer_pythorch/blob/master/changed5_img.png">
</p>


## 0. Requirements
```python
python=3.5
numpy
matplotlib
torch=1.0.0
torchvision
torchsummary
opencv-python
```

If you use google colab, you don't need to set up. Just run and run!! 



## 1. Usage
You only run `Fast-Style-Transfer-PyTorch.ipynb`. 

Or you can use Google Colab for free!! This is [colab link](https://colab.research.google).

After downloading ipynb, just upload to your google drive. and run!



## 2. Tutorial & Code implementation Blog Posting (Korean Only)
[“Fast Style Transfer PyTorch Tutorial”](https://hoya012.github.io/blog/Fast-Style-Transfer-Tutorial/)  



## 3. Dataset download 
For simplicty, i use **COCO 2017 validation set** instead of **COCO 2014 training set**.

- COCO 2014 training: about 80000 images / 13GB
- COCO 2017 validation: about 5000 images / 1GB –> i will use training epoch multiplied by 16 times

You can download COCO 2017 validation dataset in [this link](http://images.cocodataset.org/zips/val2017.zip)


## 4. Link to google drive and upload files to google drive
If you use colab, you can simply link ipynb to google drive.

```python
from google.colab import drive
drive.mount("/content/gdrive")
```

Upload COCO dataset & Style Image & Test Image or Videos to Your Google Drive.

You can use google drive location in ipynb like this codes.

```python
style_image_location = "/content/gdrive/My Drive/Colab_Notebooks/data/vikendi.jpg"

style_image_sample = Image.open(style_image_location, 'r')
display(style_image_sample)
```

## 5. Transfer learning, inference from checkpoint.
Since google colab only uses the GPU for 8 hours, we need to restart it from where it stopped.

To do this, the model can be saved as a checkpoint during training, and then the learning can be done.
Also, you can also use trained checkpoints for inferencing.

```python
transfer_learning = False # inference or training first --> False / Transfer learning --> True
ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_63_batch_id_500.pth")

if transfer_learning:
  checkpoint = torch.load(ckpt_model_path, map_location=device)
  transformer.load_state_dict(checkpoint['model_state_dict'])
  transformer.to(device)
```


## Reference
- <a href="https://github.com/pytorch/examples/tree/master/fast_neural_style" target="_blank"> pytorch example code </a>
- <a href="https://ezgif.com/video-to-gif" target="_blank"> avi --> gif for demo </a>
- <a href="https://colab.research.google.com/" target="_blank"> google colaboratory </a>
