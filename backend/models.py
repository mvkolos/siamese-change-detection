from keras.layers import Conv2D, Activation, concatenate, Input
from keras.models import Model, load_model

from segmentation_models.unet.blocks import Transpose2D_block
from segmentation_models.unet.blocks import Upsample2D_block
from segmentation_models.utils import get_layer_number, to_tuple
from segmentation_models.backbones import get_backbone
from segmentation_models.unet.model import DEFAULT_SKIP_CONNECTIONS

def build_cd_unet_sw(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):
   
    left_input = Input((None,None,3))
    right_input = Input((None,None,3))
    

    x_right = backbone(right_input)
    x_left = backbone(left_input)
    
    b1 = Model(left_input, x_left)
    b2 = Model(right_input, x_right)
    
    num_outputs = len(b1.output)
    print(num_outputs)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    x = concatenate([b1.output[0], b2.output[0]], axis=-1)
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < num_outputs-1:
            left_skip_connection = b1.output[i+1]
            right_skip_connection = b2.output[i+1]
            skip_connection = concatenate([left_skip_connection, right_skip_connection], axis=-1)

        
        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model([left_input, right_input], x)

    return model


def build_cd_unet(backbones, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):
   
    left_input = backbones[0].input
    right_input = backbones[1].input
    
    for layer in backbones[1].layers:
        layer.name = layer.name + '_right'
        
    x = concatenate([backbones[0].output, backbones[1].output], axis=-1)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbones[0], l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            left_skip_connection = backbones[0].layers[skip_connection_idx[i]].output
            right_skip_connection = backbones[1].layers[skip_connection_idx[i]].output
            skip_connection = concatenate([left_skip_connection, right_skip_connection], axis=-1)
        
        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model([left_input, right_input], x)

    return model


def cd_unet(backbone_name = 'resnet34',
            input_shape=(None, None, 3), 
           classes = 1, encoder_weights='imagenet'):
    left_encoder = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)
    
    right_encoder = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)

    skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]
    
    return build_cd_unet((left_encoder, right_encoder), classes, skip_connections)


def cd_unet_sw(backbone_name = 'resnet18',
            input_shape=(None, None, 3), 
           classes = 1, encoder_weights='imagenet'):
   
    encoder = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)


    skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]
                           
    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(encoder, l) if isinstance(l, str) else l
                               for l in skip_connections])
    activations = []

    for i in range(5):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            activations.append(encoder.layers[skip_connection_idx[i]].output)   
    
    submodel = Model(encoder.input, [encoder.output] + activations)
    
    return build_cd_unet_sw(submodel, classes, skip_connections)