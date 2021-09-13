###################################################################################################################
# ERFNet Architecture in Tensorflow                                                                               #
# Author: Maximilian Heinz (inspired by https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py)    #
#                                                                                                                 #
# The ERFNet architecture makes use of three different modules that it stacks together.                           #
#    -A factorized residual network module with dilations.                                                        #
#    -A downsampling module inspired by an inception module.                                                      #
#    -An upsampling module.                                                                                       #
#                                                                                                                 #
# The different buliding blocks/modules will be defined and then used to build the final structure                #
###################################################################################################################

#LIBRARY IMPORTS
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, MaxPooling2D, add, concatenate, Dropout, Lambda
from tensorflow.keras.models import Model

#-----------------------------------------------------------------------------------------------#
#                                    non Bottleneck Block                                       #
#   -ERFnet makes use of Resnet modules but modifies them using                                 #
#   - ReLU activation is used outside of the Conv2D function to control usage                   #
#   Factorized Convulutions and Downsampling modules (Depthwise seperable could be used instead)#
#   -Input is constructed by detecting the number of filters (last element of the shape-list),  #
#   -non-bt-1D blocks don't change the filtersize of the branch with convolutions               #
#   -The 2D kernel in the convolution Operation will be switched                                #
#   -Also the dilation rate will be adapted                                                     #
#   -Optional Usage of a kernel_initializer                                                     #
#-----------------------------------------------------------------------------------------------#

def non_bottleneck_1d(input, dilationRate, dropoutRate, name="non-bt-1d - Block"):
    #Detect number of filters (last element of the shape-list)
    n_filter = input.shape.as_list()[-1]
    conv_branch = Conv2D(
        filters=n_filter,
        kernel_size=[3,1],
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    ) (input)
    conv_branch = ReLU()(conv_branch)

    conv_branch = Conv2D(
        filters=n_filter,
        kernel_size=[1,3],                      #Change kernel to [1,3]
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    ) (conv_branch)
    conv_branch = BatchNormalization(axis=-1, epsilon=1e-03)(conv_branch)
    conv_branch = ReLU()(conv_branch)

    #Add a parameter for the dilation Rate
    conv_branch = Conv2D(
        filters=n_filter,
        kernel_size=[3,1],
        strides=1,
        dilation_rate=(dilationRate, 1),
        padding='same',
        kernel_initializer='he_normal'
    ) (conv_branch)
    conv_branch = ReLU()(conv_branch)

    conv_branch = Conv2D(
        filters=n_filter,
        kernel_size=[1,3],                      #Change kernel to [1,3]
        strides=1,
        dilation_rate=(dilationRate, 1),
        padding='same',
        kernel_initializer='he_normal'
    ) (conv_branch)
    conv_branch = BatchNormalization(axis=-1, epsilon=1e-03)(conv_branch)
    conv_branch = ReLU()(conv_branch)
    
    conv_branch = Dropout(rate=dropoutRate)(conv_branch)

    #Addition of untransformed input with convulutional branch (Residual Idea of not having to learn identity function!
    output = add([conv_branch, input])
    
    return ReLU()(output)


#-----------------------------------------------------------------------------------------------#
#                                       DownsamplerBlock                                        #
#   -Downsampling Module inspired by Inception Blocks                                           #
#   -2 branches that are concatenated in the end (MaxPoolind and Convolution)                   #
#-----------------------------------------------------------------------------------------------#

def downsampler_block(input, n_filters_out, name="Downsampling Block"):
    #We want n_filters_out at the end of this Block, but both branches are concatendated in the end!
    n_filters_in = input.shape.as_list()[-1]
    n_filters_conv = n_filters_out - n_filters_in

    branch_a = Conv2D(
        filters=n_filters_conv,
        kernel_size=[3,3],
        strides=2,
        padding='same',
        kernel_initializer='he_normal'
    ) (input)

    branch_b = MaxPooling2D(
        pool_size=(2,2),
        strides=2
    ) (input)

    #Conatenate both branches to get output with right fitler size
    output = concatenate([branch_a, branch_b])
    output = BatchNormalization(axis=-1, epsilon=1e-03)(output)

    return ReLU()(output)

#-----------------------------------------------------------------------------------------------#
#                                       UpsamplerBlock                                          #
#   -Deconvolution or inverse Convolution                                                       #
#   -Is used in the Decoder(Upsampling in steps)                                                #
#   -Doesn't include a activation layer like sigmoid or softmax -> needs to be added in model   #
#-----------------------------------------------------------------------------------------------#

def upsampler_block(input, n_filters_out, name="Upsampling Block aka Deconvolution"):
    output = Conv2DTranspose(
        filters=n_filters_out,
        kernel_size=[3,3],
        strides=2,
        padding='same',
        kernel_initializer='he_normal'
    ) (input)
    output = BatchNormalization(axis=-1, epsilon=1e-03)(output)

    return ReLU()(output)

#-----------------------------------------------------------------------------------------------#
#                                         Get Model                                             #
#   -Constructed of Encoder and Decoder                                                         #
#   -Create Input Layer based on Image Size                                                     #
#   -Last Upsampling layer to be sigmoid(binary) or softamx(multiclass) activation              #
#    (You could also connect another Conv2D layer with 1x1 kernels at the end                   #
#     but this would add additional parameters)                                                 #  
#-----------------------------------------------------------------------------------------------#

def get_erfnet(img_heigth, img_width, img_channels, num_classes = 1):
    #Define Input Layer and normalize input to be between 0 and 1
    input = Input((img_heigth, img_width, img_channels))
    normalized_input = Lambda(lambda x: x/255)(input)

    #CONSTRUCT ENCODER(1-16)

    # 2 Downsampler Blocks
    layers = downsampler_block(normalized_input, n_filters_out=16)
    layers = downsampler_block(layers, n_filters_out=64)

    # 5 non-1-bt - Blocks constant dilation rate 
    for i in range (5):
        layers = non_bottleneck_1d(
            input=layers,
            dilationRate=1,
            dropoutRate=0.03
        )

    # 1 Downsampler Block
    layers = downsampler_block(layers, n_filters_out=128)

    # 4 non-1-bt - Blocks with rising dilation rate (2,4,8,16)
    for i in range (1,5):
        layers = non_bottleneck_1d(
            input=layers,
            dilationRate=2**(i),
            dropoutRate=0.03
        )

    # 4 non-1-bt - Blocks with rising dilation rate (2,4,8,16)
    for i in range (1,5):
        layers = non_bottleneck_1d(
            input=layers,
            dilationRate=2**(i),
            dropoutRate=0.03
        )

    #CONSTRUCT DECODER (17-23)

    # 1 Upsampler Block
    layers = upsampler_block(layers, n_filters_out=64)

    # 2 non-bt-1d - Blocks with constant dilation Rate
    for i in range (2):
        layers= non_bottleneck_1d(
            input=layers,
            dilationRate=1,
            dropoutRate=0.03
        )

    layers = upsampler_block(layers, n_filters_out=16)

    #2 non-1bt-Blocks with constant dilation Rate
    for i in range (2):
        layers= non_bottleneck_1d(
            input=layers,
            dilationRate=1,
            dropoutRate=0.03
        )

    #LAYER 23 - Deconvolution with sigmoid (binary classification)!
    output = Conv2DTranspose(
        filters=num_classes,
        kernel_size=[2, 2],
        strides=2,
        padding='valid',
        activation='sigmoid',
    ) (layers)

    #Create Layers into a model
    model = Model(
        input,
        output,
    )

    #Check Architecture
    model.summary()
    tf.keras.utils.plot_model(model, "erfnet_with_shape_test.png", show_shapes=True)
    return model

#Quick Test
#get_erfnet(520,1280,3)