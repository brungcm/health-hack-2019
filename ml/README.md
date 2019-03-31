# Machine Learning models
Here it is the code responsible for training the machine learning model for drowning classification

## Extract video frames

```bash
python -m ml.extract_frames \
--config-file ./ml/config/tracker_config.json \
--video-path /dataset/hacka_videos/IMG_9016.MOV \
--output-dir /dataset/frames/train/normal/IMG_9016
```

## Generate TF-records

```bash
python -m ml.generate_tfrecords \
--input-dir /dataset/frames/train/drowning \
--output-dir /dataset/tfrecords \
--phase train \
--target drowning \
--window-size 16 \
--window-offset 8
```

```bash
python -m ml.generate_tfrecords \
--input-dir /dataset/frames/test/drowning \
--output-dir /dataset/tfrecords \
--phase test \
--target drowning \
--window-size 16 \
--window-offset 8
```

```bash
python -m ml.generate_tfrecords \
--input-dir /dataset/frames/train/normal \
--output-dir /dataset/tfrecords \
--phase train \
--target normal \
--window-size 16 \
--window-offset 8
```

```bash
python -m ml.generate_tfrecords \
--input-dir /dataset/frames/test/normal \
--output-dir /dataset/tfrecords \
--phase test \
--target normal \
--window-size 16 \
--window-offset 8
```


## Train network

```bash
python train_net.py \
--train-tf-list /dataset/tfrecords/train_normal.tfrecord.gz,/dataset/tfrecords/train_drowning.tfrecord.gz \
--test-tf-list /dataset/tfrecords/test_normal.tfrecord.gz,/dataset/tfrecords/test_drowning.tfrecord.gz \
--output-dir /models/ \
--window-size 16
```