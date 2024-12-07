import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

# Define metrics for tracking
class PSNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name="psnr", **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name="psnr", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self.psnr.assign(tf.reduce_mean(psnr))

    def result(self):
        return self.psnr

    def reset_states(self):
        self.psnr.assign(0.0)

class SSIMMetric(tf.keras.metrics.Metric):
    def __init__(self, name="ssim", **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name="ssim", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self.ssim.assign(tf.reduce_mean(ssim))

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0.0)

# Compile function
def compile_model(model):
    optimizer = Adam(learning_rate=1e-4)
    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = [
        PSNRMetric(),
        SSIMMetric(),
        MeanAbsoluteError(name="mae"),
        MeanSquaredError(name="mse")
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Build and compile the RDN models
# For Denoising
rdn_denoising_model = build_rdn_denoising(input_shape=(64, 64, 1))
compile_model(rdn_denoising_model)

# For Super-Resolution or Denoising (Generic RDN with scale factor 1)
rdn_model = build_rdn(scale_factor=1, input_shape=(64, 64, 1))
compile_model(rdn_model)

# Summary for verification
print("RDN Denoising Model:")
rdn_denoising_model.summary()
print("\nRDN Model:")
rdn_model.summary()
