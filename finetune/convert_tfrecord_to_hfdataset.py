import os
import tensorflow as tf
from datasets import load_dataset
from datasets import Dataset, Features, Value, DatasetDict


def decode_tfrecord(example_proto):
    # Define the feature description to parse the tf.train.Example messages
    feature_description = {
        'targets': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def tfrecord_to_datasets(tfrecord_files):
    # Specify the structure of the features
    features = Features({
        'targets': Value('string'),
        'id': Value('string'),
    })

    # Load the TFRecord Dataset
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(lambda x: decode_tfrecord(x))

    # Convert the TF dataset to a list of dictionaries
    dataset_dicts = []
    for parsed_record in parsed_dataset:
        dataset_dicts.append({
            "targets": parsed_record['targets'].numpy().decode('utf-8'),
            "id": parsed_record['id'].numpy().decode('utf-8'),
        })

    hf_dataset = Dataset.from_list(dataset_dicts)
    return hf_dataset


# Save hf_dataset to a JSON file
hf_dataset = tfrecord_to_datasets([os.getenv("TF_DATA_PATH")])
json_path = os.getenv("DATA_PATH")
hf_dataset.to_json(json_path)

