"""Extract parameters of the DeepLab-LargeFOV model
   from the provided .caffemodel file.
   
This scripts extracts and saves the network skeleton 
with names and shape of the parameters, 
as well as all the corresponding weights.

To run the script, PyCaffe should be installed.
"""

from __future__ import print_function

import argparse
import os
import sys

from six.moves import cPickle

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract model parameters of DeepLab-LargeFOV from the provided .caffemodel.")
    parser.add_argument("caffemodel", type=str,
                        help="Caffemodel from which the parameters will be extracted.")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Whether to store the network skeleton and weights.")
    parser.add_argument("--pycaffe_path", type=str, default="",
                        help="Path to PyCaffe (e.g., 'CAFFE_ROOT/python').")
    return parser.parse_args()

def main():
    """Extract and save network skeleton with the corresponding weights.
    
    Raises:
      ImportError: PyCaffe module is not found."""
    args = get_arguments()
    sys.path.append(args.pycaffe_path)
    try:
        import caffe
    except ImportError:
        raise
    # Load net definition.
    net = caffe.Net('./util/deploy.prototxt', args.caffemodel, caffe.TEST)
    
    # Check the existence of output_dir.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Net skeleton with parameters names and shapes.
    # In TF, the filter shape is as follows: [ks, ks, input_channels, output_channels],
    # while in Caffe it looks like this: [output_channels, input_channels, ks, ks].
    net_skeleton = list() 
    for name, item in net.params.iteritems():
        net_skeleton.append([name + '/w', item[0].data.shape[::-1]]) # See the explanataion on filter formats above.
        net_skeleton.append([name + '/b', item[1].data.shape])
    
    with open(os.path.join(args.output_dir, 'net_skeleton.ckpt'), 'wb') as f:
        cPickle.dump(net_skeleton, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Net weights. 
    net_weights = dict()
    for name, item in net.params.iteritems():
        net_weights[name + '/w'] = item[0].data.transpose(2, 3, 1, 0) # See the explanation on filter formats above.
        net_weights[name + '/b'] = item[1].data
    with open(os.path.join(args.output_dir,'net_weights.ckpt'), 'wb') as f:
        cPickle.dump(net_weights, f, protocol=cPickle.HIGHEST_PROTOCOL)
    del net, net_skeleton, net_weights

if __name__ == '__main__':
    main()