## Demo
![Demo](https://cloud.githubusercontent.com/assets/1458656/26531773/8c3eac16-43f0-11e7-8254-6b01f5f8a3db.gif)

## Install


## Generate images
``` python
python generate_image.py -f data -t p
python generate_image.py -f data -t r
python generate_image.py -f data -t s
```

## Train

### Simple CNN
``` python
python train_simple_cnn.py
```

### Use pretrained VGG
``` python
python train_bottleneck_vgg.py
```

## Classification
``` python
python classification.py -type simple
```
or

``` python
python classification.py -type vgg
```

vgg has better performance