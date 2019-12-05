# fast-style-transfer-tutorial-pytorch
Simple Tutorials &amp; Code Implementation of fast-style-transfer(Perceptual Losses for Real-Time Style Transfer and Super-Resolution, 2016 ECCV) using PyTorch. This code is based on [pytorch example codes](https://github.com/pytorch/examples/tree/master/fast_neural_style)

### Style Image from Battle Ground Game
<p align="center">
  <img width="700" src="https://github.com/hoya012/hoya012.github.io/blob/master/assets/img/fast_style_transfer/3.PNG">
</p>

### Style Transfer Demo video (Left: original / Right: output)
<p align="center">
  <img width="700" src="https://github.com/hoya012/hoya012.github.io/blob/master/assets/img/fast_style_transfer/mirama_demo.gif">
</p>

<p align="center">
  <img width="700" src="https://github.com/hoya012/hoya012.github.io/blob/master/assets/img/fast_style_transfer/sanok_demo.gif">
</p>

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

*Last update : 2019/03/05*

## Contributor
* hoya012

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

##

## 1. Usage
You only run `Fast-Style-Transfer-PyTorch.ipynb`. 

Or you can use Google Colab for free!! This is [colab link](https://colab.research.google).

After downloading ipynb, just upload to your google drive. and run!

##

## 2. Tutorial & Code implementation Blog Posting (Korean Only)
[“Fast Style Transfer PyTorch Tutorial”](https://hoya012.github.io/blog/Fast-Style-Transfer-Tutorial/)  

##

## 3. Dataset download 
For simplicty, i use **COCO 2017 validation set** instead of **COCO 2014 training set**.

- COCO 2014 training: about 80000 images / 13GB
- COCO 2017 validation: about 5000 images / 1GB –> i will use training epoch multiplied by 16 times

You can download COCO 2017 validation dataset in [this link](http://images.cocodataset.org/zips/val2017.zip)

##

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

##

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

##

## 6. Training phase

```python
if running_option == "training":
  if transfer_learning:
      transfer_learning_epoch = checkpoint['epoch'] 
  else:
      transfer_learning_epoch = 0

  for epoch in range(transfer_learning_epoch, num_epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_id + 1) + ".pth"
                print(str(epoch), "th checkpoint is saved!")
                ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
                torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
                }, ckpt_model_path)

                transformer.to(device).train()  
```

##

## 7. Test(Inference) Phase
I use video for demo. But you can use only single image. ( `running_option == "test"` )
The code below shows how to apply a style transfer with video as input and save the video as output.

If you download [trained weight](https://github.com/hoya012/fast-style-transfer-tutorial-pytorch/blob/master/ckpt_epoch_63_batch_id_500.pth), you can test without any training!

```python
if running_option == "test_video":
    
    with torch.no_grad():
        style_model = TransformerNet()

        ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_63_batch_id_500.pth")
        checkpoint = torch.load(ckpt_model_path, map_location=device)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(checkpoint.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del checkpoint[k]

        style_model.load_state_dict(checkpoint['model_state_dict'])
        style_model.to(device)

        cap = cv2.VideoCapture("/content/gdrive/My Drive/Colab_Notebooks/data/mirama_demo.mp4")

        frame_cnt = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #cv2.VideoWriter_fourcc(*'MP42')
        out = cv2.VideoWriter('/content/gdrive/My Drive/Colab_Notebooks/data/mirama_demo_result.avi', fourcc, 60.0, (1920,1080))

        while(cap.isOpened()):
            ret, frame = cap.read()
            
            try:
              frame = frame[:,:,::-1] - np.zeros_like(frame)
            except:
              break
              
            print(frame_cnt, "th frame is loaded!")

            content_image = frame
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0).to(device)

            output = style_model(content_image).cpu()
            #save_image("/content/gdrive/My Drive/Colab_Notebooks/data/vikendi_video_result/" + str(frame_cnt) +".png", output[0])
            out.write(post_process_image(output[0]))
            frame_cnt += 1
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
```

##

## Reference
- <a href="https://github.com/pytorch/examples/tree/master/fast_neural_style" target="_blank"> pytorch example code </a>
- <a href="https://ezgif.com/video-to-gif" target="_blank"> avi --> gif for demo </a>
- <a href="https://colab.research.google.com/" target="_blank"> google colaboratory </a>
