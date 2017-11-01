

from caffe2.python import brew


class ResNet20Builder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(self, model, prev_blob, no_bias=False, is_test=False, spatial_bn_mom=0.9):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.no_bias = 1 if no_bias else 0

    def add_conv(self, in_filters, out_filters, kernel, stride=1, pad=0):
        self.comp_idx += 1
        self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=self.no_bias,
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            'comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx),
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        return self.prev_blob

    def add_simple_block(
        self,
        input_filters,
        num_filters,
        down_sampling=False,
        spatial_batch_norm=True
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 3x3
        self.add_conv(
            input_filters,
            num_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(num_filters, num_filters, kernel=3, pad=1)
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
                input_filters,
                num_filters,
                weight_init=("MSRAFill", {}),
                kernel=1,
                stride=(1 if down_sampling is False else 2),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model, [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1


def create_resnet20(
    model, data, num_input_channels=3, num_labels=10, is_test=False
):
    '''
    Create residual net for cifar-10/100
    num_groups = 3
    '''
    # conv1 + maxpool, input=32x32x3, output=32x32x16
    brew.conv(model, data, 'conv1', num_input_channels, 16, kernel=3, stride=1)
    brew.spatial_bn(model, 'conv1', 'conv1_spatbn', 16, epsilon=1e-3, is_test=is_test)
    brew.relu(model, 'conv1_spatbn', 'relu1')

    builder = ResNet20Builder(model, 'relu1', is_test=is_test)

    # conv2, output=32x32x16
    builder.add_simple_block(16, 16)
    builder.add_simple_block(16, 16)
    builder.add_simple_block(16, 16)

    #conv3, output=16x16x32
    builder.add_simple_block(16, 32, down_sampling=True)
    builder.add_simple_block(32, 32)
    builder.add_simple_block(32, 32)

    #conv4, output=8x8x64
    builder.add_simple_block(32, 64, down_sampling=True)
    builder.add_simple_block(64, 64)
    builder.add_simple_block(64, 64)

    # avg_pool output=1x1, 64
    brew.average_pool(model, builder.prev_blob, 'final_avg', kernel=8, stride=1)

    # fc layer output=1x1x(num_labels)
    brew.fc(model, 'final_avg', 'last_out', 64, num_labels)
    softmax = brew.softmax(model, 'last_out', 'softmax')
    return softmax

