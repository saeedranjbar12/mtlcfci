'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
#saeed
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import torch
import torch.nn.functional as F
import scipy.stats as st


import scipy.misc
from math import sqrt
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import  matplotlib
import matplotlib.pyplot as plt
import scipy.misc
from math import sqrt
from PIL import Image
from torch.autograd import Variable
device = torch.device("cuda")

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - float(iter)/max_iter)**power


    #if iter % 10 ==1:
    #    print 'The Updated LR is {}'.format(init_lr*(1 - float(iter)/max_iter)**power)

    if iter % lr_decay_iter or iter > max_iter:
        return optimizer, float(init_lr*(1 - float(iter)/max_iter)**power)


    return optimizer,float(init_lr*(1 - float(iter)/max_iter)**power)

def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

def dump_feature_2D(feature, filename, max_range, min_range,png_jpg_flag):
    # PNG JPG FLAG , 0: PNG, 80:85:90:95 JPG
    if png_jpg_flag == 0:
        filename_write = filename + '.png'
    else:
        filename_write = filename + '.jpg'

    filename_bin = filename + '.bin'
    _, fch, frow, fcol = feature.shape
    # print (fch,frow,fcol)
    feature_vector = np.reshape(feature, (1, -1))
    # print (feature_vector.shape)
    feature_2D_height = int((2 ** np.floor(1 / 2 * (np.log2(fch)))) * frow)
    feature_2D_width = int((2 ** np.ceil(1 / 2 * (np.log2(fch)))) * fcol)
    # print (feature_2D_height,feature_2D_width)
    # feature_2D = np.reshape(feature_vector, (feature_2D_height,feature_2D_width))
    # normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
    counter = 0
    feature_2D = np.zeros((feature_2D_height, feature_2D_width))
    for i in range(0, feature_2D_height, frow):
        for j in range(0, feature_2D_width, fcol):

            for k in range(i, i + frow):
                for l in range(j, j + fcol):
                    feature_2D[k][l] = feature_vector[0][counter]
                    counter += 1

    # normalized = (feature_2D-np.min(feature_2D))/float((np.max(feature_2D)-np.min(feature_2D)))
    # print (normalized.shape)

    if png_jpg_flag == 0:
        scipy.misc.imsave(filename_write, feature_2D)  #use different libraries for saving PNG/JPEG images
    else:
        obj = Image.fromarray(feature_2D)
        obj = obj.convert("L")
        obj.save(filename_write, format='JPEG', quality=png_jpg_flag)

    # read_feature_2D = np.zeros((feature_2D_height,feature_2D_width))
    read_feature_2D = scipy.misc.imread(filename_write)
    # print (read_feature_2D.shape)
    # print (np.max(np.abs(read_feature_2D-feature_2D)))
    print (str(os.path.getsize(filename_write)))  #THIS GIVES IN BYTES!!!!!
    # plt.imshow(feature_2D,cmap=plt.cm.gray) #
    # plt.savefig('dumped_feature_2D.png',dpi=300)
    # input("wait here")

    #features_val_numpy_vector = np.reshape(feature_2D, (1, -1))
    #features_val_numpy_vector.astype('uint8').tofile(filename_bin)

    # change the image to tensor!
    channel_counter = -1
    # print (feature.shape)
    read_3D_feature = np.zeros(feature.shape)
    # print (read_3D_feature.shape)
    for i in range(0, feature_2D_height, frow):
        for j in range(0, feature_2D_width, fcol):
            channel_counter += 1
            for k in range(i, i + frow):
                for l in range(j, j + fcol):
                    read_3D_feature[0][channel_counter][k - i][l - j] = read_feature_2D[k][l]
    read_3D_feature = (read_3D_feature * (max_range - min_range) / 255.0) + min_range
    return Variable(torch.Tensor(read_3D_feature))

def Quantize_center(center_feature, n_bits=8):
    # r1 = -0.5
    # r2 = 0.5
    r1 = -0.002
    r2 = 0.002  # 2*0.002*256 = 1 it is the change in the 1/255 not 255 scale
    data_shape = center_feature.data.shape
    quant_error_t = (r1 - r2) * torch.rand(data_shape) + r2

    # quant_error = Variable(quant_error.cuda(), requires_grad=True)
    quant_error = quant_error_t.to(device)
    # quant_error.cuda()
    center_recon = center_feature + quant_error
    return center_recon

def Quantize_255(center_feature, n_bits=8):  # I consider numbers to be 255
    # data_shape = center_feature.data.shape
    center_flatten = center_feature.view(center_feature.numel())
    max_center_value, _ = center_flatten.max(0)
    min_center_value, _ = center_flatten.min(0)
    max_value_return = max_center_value.data.cpu().numpy()
    min_value_return = min_center_value.data.cpu().numpy()
    if torch.eq(min_center_value, max_center_value).all():
        max_center_value = max_center_value.add(10 ^ (-10))
    V_minus_min_v = center_feature - min_center_value.expand_as(center_feature)
    range_center = max_center_value.add(-1 * min_center_value)
    # print range_center
    range_center_inv = range_center.pow(-1)
    bit_range = 2 ** (n_bits) - 1
    range_center_inv_bit_range = bit_range * range_center_inv
    value_before_round = V_minus_min_v * range_center_inv_bit_range.expand_as(V_minus_min_v)

    return _, value_before_round  # quantized_value_255

def Quantize_center_VALIDATION(center_feature, n_bits=8):  # I consider numbers to be 255
    # data_shape = center_feature.data.shape
    center_flatten = center_feature.view(center_feature.numel())
    max_center_value, _ = center_flatten.max(0)
    min_center_value, _ = center_flatten.min(0)
    max_value_return = max_center_value.data.cpu().numpy()
    min_value_return = min_center_value.data.cpu().numpy()

    if torch.eq(min_center_value, max_center_value).all():
        max_center_value = max_center_value.add(10 ^ (-20)) #to avoid zero range
    V_minus_min_v = center_feature - min_center_value.expand_as(center_feature)
    range_center = max_center_value.add(-1 * min_center_value)
    range_center_inv = range_center.pow(-1)
    bit_range = 2 ** (n_bits) - 1
    range_center_inv_bit_range = bit_range * range_center_inv
    value_before_round = V_minus_min_v * range_center_inv_bit_range.expand_as(V_minus_min_v)
    quantized_value_255 = value_before_round.round()
    # Now we go all the way back to get the original value
    range_center_bit_range_inv = range_center_inv_bit_range.pow(-1)
    orig_before_add_min = quantized_value_255 * range_center_bit_range_inv.expand_as(quantized_value_255)
    center_recon = orig_before_add_min + min_center_value.expand_as(orig_before_add_min)
    return center_recon, quantized_value_255, max_value_return, min_value_return

