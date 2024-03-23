import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def create_tf_example(target, id_value):
    feature = {
        'targets': _bytes_feature(target),
        'id': _bytes_feature(id_value),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecords(file_name, num_examples=30000):
    with tf.io.TFRecordWriter(file_name) as writer:
        for i in range(num_examples):
            target = "# In[ ]:\n\n\nimport pandas as pd\nimport numpy as np\n\n\n\n# In[ ]:\n\n\ndf = pd.read_csv('load_data.csv')"
            id_value = f"notebook_{i}|||turn_0"
            tf_example = create_tf_example(target, id_value)
            writer.write(tf_example.SerializeToString())

# Specify the TFRecord file name
tfrecord_file_name = 'gift4code.tfrecord'

# Write 100 examples to the TFRecord file
write_tfrecords(tfrecord_file_name)
