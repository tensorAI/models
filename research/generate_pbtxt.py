import csv
import argparse, os, sys

def load_categories_from_csv_file(args):
  """Loads categories from a csv file.
  The CSV file should have one comma delimited numeric category id and string
  category name pair per line. For example:
  0,"cat"
  1,"dog"
  2,"bird"
  ...
  Args:
    csv_path: Path to the csv file to be parsed into categories.
  Returns:
    categories: A list of dictionaries representing all possible categories.
                The categories will contain an integer 'id' field and a string
                'name' field.
  Raises:
    ValueError: If the csv file is incorrectly formatted.
  """
  categories = []

  with open(args.label_map_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i,row in enumerate(reader):
      if not row:
        continue

      if len(row) != 2:
        raise ValueError('Expected 2 fields per row in csv: %s' % ','.join(row))

      category_id = int(i+1)
      category_name = row[0]
      category_display_name = row[1]
      categories.append({'id': category_id, 'name': category_name, 'display_name': category_display_name})

    return categories

def main(args):

    """saves the dict to .pbtxt file

    """
    # load the categories
    categories = load_categories_from_csv_file(args)

    #make pbtxt file
    categories.sort(key=lambda x: x['id'])
    with open(args.label_map_file, 'wb+') as f:
        count_sorted = [(cat['name'], cat['display_name']) for cat in categories]
        for i, class_name in enumerate(count_sorted):
            f.write("item {\n")
            f.write("    id: {}\n".format(i+1))
            f.write("    name: \'{}\'\n".format(class_name[0]))
            f.write("    display_name: \'{}\'\n".format(class_name[1]))
            f.write("}\n\n")
    return None

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_map_csv_path',
                        type=str,
                        help='label map csv to read',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset/class-descriptions-boxable.csv',
                        required=False)
    parser.add_argument('--label_map_file',
                        type=str,
                        help='label map file to write',
                        default='/media/jay/data/Dataset/object_detection/open_image_dataset/label_map.pbtxt',
                        required=False)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))