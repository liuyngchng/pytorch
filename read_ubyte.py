#!/usr/bin/python3
import struct


def read_images(file_path):
    file = open(file_path, 'rb')
    magic_number, num_images, num_rows, num_columns = struct.unpack('>IIII', file.read(16))
    print("magic_number, num_images, num_rows, num_columns = {}, {}, {}, {}"
          .format(magic_number, num_images, num_rows, num_columns))


if __name__ == "__main__":
    my_file = "./data/FashionMNIST/raw/t10k-images-idx3-ubyte"
    print("read file {}".format(my_file))
    read_images(my_file)
