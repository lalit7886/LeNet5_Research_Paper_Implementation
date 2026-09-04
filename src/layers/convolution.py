import numpy as np
class Conv2D:
    def __init__(self, out_channels=6, in_channels=1, kernel_size=3):
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.bias = np.zeros((out_channels, 1))

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_shape = None

        self.dw_input = None
        self.dw_bias = None
        self.dw_weights = None

    def forward(self, x):
        self.input = x
        self.input_shape = x.shape

        batch_size, channels, height, width = x.shape

        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        output = np.zeros(
            (batch_size, self.out_channels, output_height, output_width)
        )

        for batch_index, batch in enumerate(x):

            for filter_id, filter_weights in enumerate(self.weights):

                for h in range(output_height):
                    for w in range(output_width):

                        input_patch = batch[
                            :,
                            h:h + self.kernel_size,
                            w:w + self.kernel_size
                        ]

                        output[batch_index, filter_id, h, w] = (
                            np.sum(filter_weights * input_patch)
                            + self.bias[filter_id, 0]
                        )

        return output

    def backward(self, d_upstream):

        batch_size, out_channels, output_height, output_width = d_upstream.shape

        _, in_channels, input_height, input_width = self.input_shape

        # Gradients
        dw_weights = np.zeros_like(self.weights)
        dw_bias = np.zeros_like(self.bias)
        dw_input = np.zeros_like(self.input)

        K = self.kernel_size

        for batch_index in range(batch_size):

            for filter_id in range(out_channels):

                # Upstream gradient for this output filter
                dout_filter = d_upstream[batch_index, filter_id]

                for h in range(output_height):
                    for w in range(output_width):

                        # Scalar upstream gradient
                        gradient = dout_filter[h, w]

                        # Corresponding input region
                        input_patch = self.input[
                            batch_index,
                            :,
                            h:h + K,
                            w:w + K
                        ]

                        # Actual CNN filter
                        filter_weights = self.weights[filter_id]

                        dw_input[
                            batch_index,
                            :,
                            h:h + K,
                            w:w + K
                        ] += gradient * filter_weights


                        dw_weights[filter_id] += gradient * input_patch

            
                        dw_bias[filter_id, 0] += gradient

        self.dw_input = dw_input
        self.dw_weights = dw_weights
        self.dw_bias = dw_bias

        return dw_input, dw_weights, dw_bias


if __name__ == "__main__":

    x = np.array([[
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    ]], dtype=float)

    conv = Conv2D(
        out_channels=1,
        in_channels=1,
        kernel_size=3
    )

    output = conv.forward(x)

    print("Input:")
    print(x)

    print("\nForward output:")
    print(output)

    d_upstream = np.ones_like(output)

    dx, dw, db = conv.backward(d_upstream)

    print("\ndX:")
    print(dx)

    print("\ndW:")
    print(dw)

    print("\ndb:")
    print(db)