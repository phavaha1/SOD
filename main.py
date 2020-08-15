from sod import SOD
from PIL import Image
import os
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help='select model name: u2net or u2netp', required=True)
    parser.add_argument('-im', '--image_name', help='select image to test', required=True)

    args = parser.parse_args()

    sod = SOD(args.model_name)

    input_image = Image.open(os.path.join(os.getcwd(), 'test', 'input_images', args.image_name))
    mask = sod.get_mask(np.array(input_image))

    mask = mask.resize((input_image.size[0], input_image.size[1]), resample=Image.BILINEAR)

    # convert to grey:
    mask = mask.convert('L')

    empty = Image.new("RGBA", input_image.size, 0)

    output_image = Image.composite(input_image, empty, mask)
    output_image.save(os.path.join(os.getcwd(), 'test', 'output_images', args.image_name.split('.')[0] + '.png'))
