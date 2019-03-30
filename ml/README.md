# Machine Learning models
Here it is the code responsible for training the machine learning model for drowning classification

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