import subprocess
import struct
from numpy import *
from matplotlib.pyplot import *
import cv2

def download(url):
    """Download a GZIP archive, return the data as a byte string."""
    # Do the download by shelling out to curl.
    cmd = 'curl "%s" | gzip -d' % url
    return subprocess.check_output(cmd, shell=True)

def get_files():
    """Download MNIST files from the internet."""
    url_format = "http://yann.lecun.com/exdb/mnist/%s-%s-idx%d-ubyte.gz"
    files = [("train", "images", 3), ("train", "labels", 1),
             ("t10k", "images", 3), ("t10k", "labels", 1)]

    urls = [url_format % values for values in files]
    data = [download(url) for url in urls]
    return data

data = get_files()


def parse_labels(data):
    """Parse labels from the binary file."""

    # We're going to use the Python 'struct' package.
    # This is an incredibly nice package which allows us to specify the format
    # our data is in, and then automatically parses the data from the string.
    # Let's start by getting the magic number and the length: the first character
    # represents the endianness of the data (in our case, '>' for big endian), while
    # the next characters are the format string ('2i' for two integers).
    magic, n = struct.unpack_from('>2i', data)
    assert magic == 2049, "Wrong magic number: %d" % magic

    # Next, let's extract the labels.
    labels = struct.unpack_from('>%dB' % n, data, offset=8)
    return labels

def parse_images(data):
    """Parse images from the binary file."""

    # Parse metadata.
    magic, n, rows, cols = struct.unpack_from('>4i', data)
    assert magic == 2051, "Wrong magic number: %d" % magic

    # Get all the pixel intensity values.
    num_pixels = n * rows * cols
    pixels = struct.unpack_from('>%dB' % num_pixels, data, offset=16)

    # Convert this data to a NumPy array for ease of use.
    pixels = asarray(pixels, dtype=ubyte)

    # Reshape into actual images instead of a 1-D array of pixels.
    images = pixels.reshape((n, cols, rows))
    return images

train_images = parse_images(data[0])
train_labels = parse_labels(data[1])
print train_images[0].size
test_images = parse_images(data[2])
test_labels = parse_labels(data[3])



# Get the figure and axes.
fig, axes = subplots(5, 5)
axes = axes.reshape(25)
fig.suptitle("Random Sampling of MNIST")

# Plot random images.
indices = random.randint(len(train_images), size=25)
for axis, index in zip(axes, indices):
    image = train_images[index, :, :]
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.imshow(image, cmap = cm.Greys_r)
    cv2.imshow('image',image)
    cv2.waitKey(0)
