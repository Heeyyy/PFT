def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def getSaliencyMap(image):
    """
    
    GET SaliencyMap using PFT method
    using FFT get gray image's phase spectrum
    then IFFT phase spectrum get SaliencyMap,
    finally normlization SaliencyMap to 0 or 1
    
    input ： image shape == H*W*3
    output ：Saliency Map shape==H*W*1
    
    """
    a_gray = tf.cast(tf.image.rgb_to_grayscale(image), tf.complex64)
    gauss_kernel = gaussian_kernel(7, 0., 1.)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

    a_fft = tf.fft2d(a_gray)
    phase = tf.angle(a_fft)
    phase = tf.complex(real=tf.math.cos(phase), imag=tf.math.sin(phase))
    s = tf.ifft2d(phase)
    salientmap = tf.pow(tf.abs(s), 2)
    salientmap = tf.expand_dims(salientmap, axis=0)
    salientmap = tf.nn.conv2d(salientmap, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    maxval = tf.reduce_max(salientmap)
    minval = tf.reduce_min(salientmap)
    scale = 255 / (maxval - minval)
    salientmap = ((salientmap - minval) * scale)
    mean = tf.reduce_mean(salientmap)
    salientmap = salientmap - mean
    SaliencyMap = (tf.sign(salientmap) + 1)/2
    SaliencyMap=tf.squeeze(SaliencyMap,axis=0)
    print(SaliencyMap)   
    return SaliencyMap
