import shutil

source_dir = '/home/ustcgmh/style-transfer/train2014_min'
output_filename = '/home/ustcgmh/style-transfer/train2014_min'

shutil.make_archive(output_filename, 'zip', source_dir)