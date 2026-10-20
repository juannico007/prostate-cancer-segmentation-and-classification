import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crop the encoder feature map to match the decoder feature map size.

    Parameters:
        encoder_layer: Feature map from the encoder path.
        decoder_layer: Feature map from the decoder path.

    Returns:
        Tuple: Cropped encoder and original decoder tensors.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer

def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True
):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def get_up_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    up_mode: str = 'transposed'
):
    if up_mode == 'transposed':
        return nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)

def get_maxpool_layer(
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0
):
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

def get_activation(activation: str = 'relu'):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=1)
    else:
        raise ValueError(
            f'Unknown activation type "{activation}".\n'
            'Valid choices are "relu", "leaky", "elu" or "prelu"'
        )
    
def get_normalization(
    num_channels: int,
    normalization: str = 'batch'
):
    if normalization.startswith('group'):
        if len(normalization) > len('group') and normalization[len('group'):].isdigit():
            num_groups = int(normalization[len('group'):])
        else:
            raise ValueError(
            f'Unknown normalization type "{normalization}".\n'
            'Valid format is group<N> where n is number of groups'
        )
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm3d(num_channels)
    elif normalization == 'batch':
        return nn.BatchNorm3d(num_channels)
    else:
        raise ValueError(
            f'Unknown normalization type "{normalization}".\n'
            'Valid choices are "group<N>", "instance" or "batch"'
        )

class Concatenate(nn.Module):
    """
    Concatenate two tensors along the channel dimension.
    """
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x

class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.sub_sample_factor = sub_sample_factor
        self.sub_sample_kernel_size = sub_sample_factor
        self.inter_channels = inter_channels
        self.upsample_mode = 'trilinear'
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
                
        self.W = nn.Sequential(
            nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm3d(self.in_channels)
        )
        
        self.theta = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=self.sub_sample_kernel_size,
                stride=self.sub_sample_factor,
                padding=0,
                bias=False
            )
        
        self.phi = nn.Conv3d(
                in_channels=self.gating_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        
        self.psi = nn.Conv3d(
                in_channels=self.inter_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        
        self.init_weights()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:], mode=self.upsample_mode, align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x.shape[2:], mode=self.upsample_mode, align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        wy = self.W(y)

        return wy, sigm_psi_f
    
    def init_weights(self):
            def weight_init(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif classname.find('Linear') != -1:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif classname.find('BatchNorm') != -1:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
            self.apply(weight_init)

class FillerBlock(nn.Module):
    def forward(self, x, g):
        return x, None

class DownBlock(nn.Module):
    """
    Downsampling block with two convolutional layers and optional max pooling.

    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        pooling: Whether to apply max pooling
        activation: Activation function name
        normalization: Normalization type
        conv_mode: 'same' or 'valid' padding

    Returns:
        output_after_pooling, output_before_pooling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: bool = True,
        activation: str = 'relu',
        normalization: str = 'batch',
        conv_mode: str = 'same'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.activation = activation
        self.normalization = normalization
        padding = 1 if 'same' in conv_mode else 0

        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=padding, bias=True)
        self.act1 = get_activation(self.activation)
        
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=padding, bias=True)
        self.act2 = get_activation(self.activation)

        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0)

        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        y = self.conv2(y)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)

        before_pool = y 
        if self.pooling:
            y = self.pool(y)
        return y, before_pool

class UpBlock(nn.Module):
    """
    Upsampling block with two convolutional layers and one upsample layer.

    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function name
        normalization: Normalization type
        conv_mode: 'same' or 'valid' padding
        up_mode: Upsampling type ('transposed' or interpolation mode)

    Returns:
        Upsampled feature map.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        normalization: str = 'batch',
        conv_mode: str = 'same',
        up_mode: str = 'transposed',
        attention: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.normalization = normalization
        padding = 1 if 'same' in conv_mode else 0
        self.up_mode = up_mode

        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, up_mode=self.up_mode)

        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=padding,bias=True)
        self.conv3 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=padding, bias=True)

        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)
        self.act3 = get_activation(self.activation)

        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm3 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)

            
        if attention:
            self.attention = GridAttentionBlock(in_channels=in_channels // 2, gating_channels=in_channels)
        else:
            self.attention = FillerBlock()

        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        up_layer = self.up(decoder_layer)
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)
        #TODO implement the attention in the forward pass
        gated_encoder, att = self.attention(cropped_encoder_layer, decoder_layer)

        if self.up_mode != 'transposed':
            up_layer = self.conv1(up_layer)
        up_layer = self.act1(up_layer)
        if self.normalization:
            up_layer = self.norm1(up_layer)

        merged_layer = self.concat(up_layer, gated_encoder)
        y = self.conv2(merged_layer)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)
        y = self.conv3(y)
        y = self.act3(y)
        if self.normalization:
            y = self.norm3(y)
        return y

class ClassifierBlock(nn.Module):
    """
    Classifier block with fully layers outputing n binary classes.

    Parameters:
        in_channels: Number of input channels
        middle_neurons: Number of neurons in the middle layer
        out_neurons: Number of output neurons
        drop: Dropout rate in the middle layer
        activation: Activation function name
    Returns:
        classification output.
    """

    def __init__(
        self,
        in_channels: int,
        out_neurons: int,
        middle_neurons: int,
        drop: float,
        activation: str = 'relu',
        ordinal: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.middle_neurons = middle_neurons
        self.out_neurons = out_neurons
        self.drop = drop
        self.activation = activation
        self.ordinal = ordinal
    
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(self.in_channels, self.middle_neurons)
        self.act1 = get_activation(self.activation)
        self.dropout = nn.Dropout(self.drop)
        self.fc2 = nn.Linear(
            self.middle_neurons,
            1 if self.ordinal else self.out_neurons,
            bias=not self.ordinal,
        )
        if self.ordinal:
            self.ordinal_bias = nn.Parameter(torch.zeros(self.out_neurons))
        
    def initialize_classifier(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.ordinal:
            nn.init.zeros_(self.ordinal_bias)

    def forward(self, encoded_features):
        y = self.gap(encoded_features)
        y = y.view(y.size(0), -1)

        y = self.fc1(y)
        y = self.act1(y)
        y = self.dropout(y)
        y = self.fc2(y)
        if self.ordinal:
            y = y + self.ordinal_bias
        return y

class UNet(nn.Module):
    """
    3D U-Net implementation.

    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_blocks: Number of encoder/decoder levels
        start_filters: Number of filters in the first layer
        activation: Activation function name. Defaults to 'relu'
        normalization: Normalization type. Defaults to 'batch'
        conv_mode: 'same' or 'valid'. Defaults to 'same'
        up_mode: 'transposed' or interpolation mode. Defaults to 'transposed'.
        middle_neurons: number of middle neurons for classifier block
        class_output: number of output neurons in the classifier
        dropout: dropout rate in the classifier
    Returns:
        Final output feature map.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 4,
        n_blocks: int = 4,
        start_filters: int = 32,
        activation: str = 'relu',
        normalization: str = 'batch',
        conv_mode: str = 'same',
        up_mode: str = 'transposed',
        middle_neurons: int = 256,
        dropout: float = 0,
        attention: bool = False,
        num_severity_levels: int = 5,
        ordinal_classification: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.up_mode = up_mode
        self.middle_neurons = middle_neurons
        self.dropout = dropout
        self.attention = attention
        self.num_severity_levels = num_severity_levels
        self.ordinal_classification = ordinal_classification
        if self.num_severity_levels < 2:
            raise ValueError("num_severity_levels must be at least 2.")

        self.down_blocks = []
        self.up_blocks = []
        self.layer_activations = []

        # Encoder
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(
                in_channels=num_filters_in,
                out_channels=num_filters_out,
                pooling=pooling,
                activation=self.activation,
                normalization=self.normalization,
                conv_mode=self.conv_mode
            )

            self.down_blocks.append(down_block)

        # Decoder
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(
                in_channels=num_filters_in,
                out_channels=num_filters_out,
                activation=self.activation,
                normalization=self.normalization,
                conv_mode=self.conv_mode,
                up_mode=self.up_mode,
                attention=self.attention
            )

            self.up_blocks.append(up_block)

        bottleneck_channels = self.start_filters * (2 ** (self.n_blocks - 1))
        self.segmentation_head = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, padding=0)
        self.severity_head = ClassifierBlock(in_channels=bottleneck_channels,
                                             middle_neurons=self.middle_neurons,
                                             out_neurons=(
                                                 self.num_severity_levels - 1
                                                 if self.ordinal_classification
                                                 else self.num_severity_levels
                                             ),
                                             drop=self.dropout,
                                             ordinal=self.ordinal_classification)
        self.severity_head.initialize_classifier()
        
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, GridAttentionBlock):
            return
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)) and getattr(module, 'weight') is not None:
            method(module.weight, **kwargs)

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, GridAttentionBlock):
            return
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)) and getattr(module, 'bias') is not None:
            method(module.bias, **kwargs)

    def initialize_parameters(
        self,
        method_weights=nn.init.xavier_uniform_,
        method_bias=nn.init.zeros_,
        kwargs_weights={},
        kwargs_bias={}
    ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)
            self.bias_init(module, method_bias, **kwargs_bias)

    def forward(self, x: torch.Tensor):
        encoder_output = []
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        severity_logits = self.severity_head(x)

        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)
        
        segmentation_logits = self.segmentation_head(x)

        return {
            "segmentation_logits": segmentation_logits,
            "severity_logits": severity_logits,
        }