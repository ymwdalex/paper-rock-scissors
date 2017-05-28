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