import tensorflow as tf
import os, argparse, sys
import pandas as pd
import hashlib, io
from tqdm import tqdm
from PIL import Image
from generate_pbtxt import load_categories_from_csv_file
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(args, image_id, annotations, label_map):
    """creates tf_example for tf record writing
    args:
        args        : arguments namespace
        image_id    : unique ID to identify the image
        annotations : annotations dataframe for the given ImageID
        label_map   : label map dataframe containing all labels and mappings
    return:
        tf_example  : a populated tf_example
    """
    image_path = os.path.join(args.input_images_directory, image_id + '.jpg')
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size # Image height / # Image width
    if image.format != 'JPEG':
        tf.logging.warning('Image format not JPEG')
        return None
    key = hashlib.sha256(encoded_jpg).hexdigest()

    filename = image_id+'.jpg' # Filename of the image. Empty if image is not from file
    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = 'jpeg' # b'jpeg' or b'png'

    xmins = annotations.XMin.values.tolist() # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = annotations.XMax.values.tolist() # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = annotations.YMin.values.tolist() # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = annotations.YMax.values.tolist() # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    frame_indices = [label_map.set_index('name').index.get_loc(i) for i in annotations.LabelName.values.tolist()]
    classes_text = label_map.display_name.values[frame_indices].tolist() # List of string class name of bounding box (1 per box)
    classes = label_map.id.values[frame_indices].tolist() # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(args):

    #read all the images and generate image ids
    all_images = tf.gfile.Glob(
          os.path.join(args.input_images_directory, '*.jpg'))
    all_image_ids = ["{0:16s}".format(os.path.splitext(os.path.basename(v))[0]) for v in all_images]
    all_image_data = pd.DataFrame({'ImageID': all_image_ids, 'image_name': all_images})

    #read annotations csv
    label_map_dicts = load_categories_from_csv_file(args)
    label_map = pd.DataFrame({'id': map(lambda d: d['id'], label_map_dicts),
                              'name': map(lambda d: d['name'], label_map_dicts),
                              'display_name': map(lambda d: d['display_name'], label_map_dicts)})
    # annotations csv read
    all_box_annotations = pd.read_csv(args.annotations_csv)
    all_box_annotations = all_box_annotations.loc[all_box_annotations.ImageID.isin(all_image_ids)]

    train_writer = tf.python_io.TFRecordWriter(os.path.join(args.output_path,'oid_train.tfrecord'))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(args.output_path,'oid_test.tfrecord'))

    for count, image_data in tqdm(enumerate(all_box_annotations.groupby(['ImageID'])), desc='Building tf records'):
        image_id, annotations = image_data
        tf_example = create_tf_example(args, image_id, annotations, label_map)
        if tf_example is not None:
            if count < int(args.frac * len(all_images)):
                train_writer.write(tf_example.SerializeToString())
            else:
                test_writer.write(tf_example.SerializeToString())
    train_writer.close()
    test_writer.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_csv',
                        type=str,
                        help='train - annotations - bbox.csv',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset/train-annotations-bbox.csv',
                        required=False)

    parser.add_argument('--boxable_csv',
                        type=str,
                        help='read train-images-boxable.csv',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset/train-images-boxable.csv',
                        required=False)

    parser.add_argument('--label_map_csv_path',
                        type=str,
                        help='label map csv to read',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset/class-descriptions-boxable.csv',
                        required=False)

    parser.add_argument('--input_images_directory',
                        type=str,
                        help='dataset csv after merging other dataset csvs',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset/images/train_00',
                        required=False)

    parser.add_argument('--frac',
                        type=float,
                        help='fraction for training and test split',
                        default=0.7,
                        required=False)

    parser.add_argument('--output_path',
                        type=str,
                        help='Path to output TFRecord',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset',
                        required=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    df = main(parse_arguments(sys.argv[1:]))
