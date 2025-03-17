
"""

  TASK B: 
  Create an own function for the convolution operation
  and convolve the filter w with the image x. 
  Use a stride of 1 and zero-pad the image such that the spatial 
  resolution of the image stays the same after the operation.
  
  INFO:
  The convolution operation involves sliding a filter w over the image x and computing 
  element-wise multiplications followed by summation. You need to implement
  this manually.
  
"""


from skimage.data import camera
import numpy as np

def convolution(image, filter, stride=1, padding=0):
    """
    Manually performs a convolution operation.
    
    Args:
    - image: numpy array of shape (N, Cin, H, W) representing the input image.
    - filter: numpy array of shape (Cout, Cin, Fh, Fw) representing the filters.
    - stride: int, step size for the convolution.
    - padding: int, amount of zero-padding to add to the image.
    
    Returns:
    - output: numpy array of shape (N, Cout, H, W), the convolved image.
    """

    # Extract dimensions
    N, Cin, H, W = image.shape
    Cout, _, Fh, Fw = filter.shape

    # Pad the image
    image_padded = np.pad(image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Compute output dimensions (same as input due to padding)
    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1

    # Initialize output tensor
    output = np.zeros((N, Cout, H_out, W_out))

    # Perform convolution
    for n in range(N):  # Loop over batch
        for cout in range(Cout):  # Loop over output channels (number of filters)
            for h in range(H_out):
                for w in range(W_out):
                    # Extract the region of interest
                    h_start, w_start = h * stride, w * stride
                    h_end, w_end = h_start + Fh, w_start + Fw
                    region = image_padded[n, :, h_start:h_end, w_start:w_end]  # Extract patch
                    
                    # Apply the filter and sum
                    output[n, cout, h, w] = np.sum(region * filter[cout, :, :, :])

    return output


if __name__ == "__main__":
    # Load image
    x = np.array(camera()).reshape(1, 1, 512, 512)

    # Define filter (Sobel edge detection)
    filter_size = 3
    w = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])  # Shape (1, 1, 3, 3)

    # Apply zero-padding
    padding = (filter_size - 1) // 2  # Ensures same spatial dimensions
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Perform convolution
    output = convolution(image=x_padded, filter=w, stride=1, padding=padding)

    # Print results
    print("Original shape:", x.shape)  # (1, 1, 512, 512)
    print("Padded shape:", x_padded.shape)  # (1, 1, 514, 514)
    print("Output shape:", output.shape)  # Should match original shape (1, 1, 512, 512)
