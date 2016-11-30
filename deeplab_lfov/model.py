import tensorflow as tf
from six.moves import cPickle

with open("from_caffe/net_skeleton.cpkt", "rb") as f:
    net_skeleton = cPickle.load(f)

# TO DO: brush up this part
ks = 3
num_layers    = [2, 2, 3, 3, 3, 1, 1, 1]
dilations     = [[1, 1],
                 [1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [2, 2, 2],
                 [12], 
                 [1], 
                 [1]]

def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation 
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.
    
    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """
    
    def __init__(self, input_size, weights_path, enable_crf=False):
        """Create the model.
        
        Args:
          input_size: a tuple of integers, representing height and width of the input image, respectively.
          weights_path: the path to the cpkt file with dictionary of weights.
          enable_crf: if set, apply CRF during the post-processing with default parameters.
        """
        self.input_size = input_size
        self.enable_crf = enable_crf
        
        self.variables = self._create_variables(weights_path)
        
    def _create_variables(self, weights_path):
        """Create all variables used by the network.
        This allows to share them between multiple calls to the loss
        function and generation function.
        
        Args:
          weights_path: the path to the cpkt file with dictionary of weights.
        
        Returns:
          A dictionary with all variables.
        """
        var = list()
        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f)
        
                # TO DO: brush this
        index = 0
        for block_index, layers in enumerate(num_layers):
            with tf.variable_scope("block{}".format(block_index)):
                current = list()
                for _ in xrange(layers):
                    current.append(tf.Variable(weights[net_skeleton[index][0]],
                                               name=net_skeleton[index][0]
                                               )
                                   )
                    index += 1
                    current.append(tf.Variable(weights[net_skeleton[index][0]],
                                               name=net_skeleton[index][0]
                                               )
                                   )
                    index += 1
                var.append(current)
        del weights
        '''
        # TO DO: brush this
        index = 0
        for block_index, layers in enumerate(num_layers):
            with tf.variable_scope("block{}".format(block_index)):
                current = list()
                for _ in xrange(layers):
                    current.append(create_variable(net_skeleton[index][0],
                                                   list(net_skeleton[index][1])
                                                   )
                                   )
                    index += 1
                    current.append(create_variable(net_skeleton[index][0],
                                                   list(net_skeleton[index][1])
                                                   )
                                   )
                    index += 1
                var.append(current)
        '''
        return var
    
    def _create_conv_block(self, input_batch, block_index):
        """Create a single block of [conv-relu]{2,3}-pool layers.
        
        Args:
          input_batch: input to the block.
          layer_index: the index of the current block.
          
        Returns:
          An output of the block.
        """
        variables = self.variables[block_index]
        current = input_batch
    
        for l in xrange(len(variables) / 2): 
            weights_filter = variables[l * 2]
            bias_filter = variables[l * 2 + 1]
            dilation = dilations[block_index][l]
            if dilation == 1:
                conv = tf.nn.conv2d(current, weights_filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv = tf.nn.atrous_conv2d(current, weights_filter, dilation, padding='SAME')
            current = tf.nn.relu(tf.nn.bias_add(conv, bias_filter))
            
        # pooling
        if block_index < 3:
            current = tf.nn.max_pool(current, 
                                     ksize=[1, ks, ks, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
        elif block_index == 3:
            current = tf.nn.max_pool(current, 
                         ksize=[1, ks, ks, 1],
                         strides=[1, 1, 1, 1],
                         padding='SAME')
        elif block_index == 4:
            current = tf.nn.max_pool(current, 
                                     ksize=[1, ks, ks, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME')
            current = tf.nn.avg_pool(current, 
                                     ksize=[1, ks, ks, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME')
        elif block_index <= 6:
            current = tf.nn.dropout(current, keep_prob=0.5)
            
        return current
    
    def _create_network(self, input_batch):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          A downsampled segmentation mask. 
        """
        current_layer = input_batch
        for block_index in xrange(len(dilations)):
            current_layer = self._create_conv_block(current_layer, block_index)
        return current_layer
    
    def _load_weights(self, weights_path):
        """Load weights from the pre-trained network.
        
        Args:
          weights_path: the path to the cpkt file with dictionary of weights.
        """
        with open(weights_path, "rb") as f:
            weights = cPickle.load(f)
        for block in xrange(len(self.variables)):
            for variable in self.variables[block]:
                print variable.name
                variable.assign(weights[variable.name[7:-2]])
      
    def preds(self, input_batch):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          A unnormalised downsampled predictions from the network.
        """
        height, width = input_batch.shape[1:3]
        raw_output = self._create_network(input_batch)
        raw_output = tf.image.resize_images(raw_output, [height, width])
        return raw_output
        #self._load_weights("from_caffe/net_weights.cpkt") # TO DO: Should be run once, during the initialisation
