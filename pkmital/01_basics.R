# Adapted from Python tensorflow tutorials at
# https://github.com/pkmital/tensorflow_tutorials
# NB: leaves out all the plotting bits

library(tensorflow)

# First, a tf$tensor
n_values <- 32L
x <- tf$linspace(-3.0, 3.0, n_values)

# Start a tf session and execute the graph
sess <- tf$Session()   # Note capital "S"! Case sensitive.
result1 <- sess$run(x)

# Alternatively, pass a session to the "eval" function
result2 <- x$eval(session=sess)
# Note that x$eval() doesn't work because it requires a session...

# You can set up an interactive session to avoid passing the session around
sess$close()
sess <- tf$InteractiveSession()

# Now x$eval() should work!
result3 <- x$eval()

# Now try a tf$Operation...
#
# Use the values in [-3, 3] that we generated earlier to create a
# Gaussian distribution
sigma <- 1.0
mean <- 0.0
z <- (tf$exp(tf$negative(tf$pow(x - mean, 2.0) / (2.0 * tf$pow(sigma, 2.0)))) * (1.0 / (sigma * tf$sqrt(2.0 * 3.14159))))

# By default, new operations are added to the default graph
comparison_graph <- z$graph$version == tf$get_default_graph()$version
result4 <- z$eval()

# Find out the shape of a tensor
shape1 <- z$get_shape()

# And in a friendlier format...
shape2 <- z$get_shape()$as_list()

# Sometimes we may not know the shape of a tensor
# until it is computed in the graph.  In that case
# we should use the tf.shape fn, which will return a
# Tensor which can be eval'ed, rather than a discrete
# value of tf.Dimension
result5 <- tf$shape(z)$eval()

# We can combine tensors:
result6 <- tf$stack(list(tf$shape(z), tf$shape(z), shape(3L), shape(4L)))$eval()

# Multiply to get a 2D Gaussian
z_2d <- tf$matmul(tf$reshape(z, c(n_values, 1L)), tf$reshape(z, c(1L, n_values)))

# Create a Gabor filter (https://en.wikipedia.org/wiki/Gabor_filter)
x_g <- tf$reshape(tf$sin(tf$linspace(-3.0, 3.0, n_values)), c(n_values, 1L))
y_g <- tf$reshape(tf$ones_like(x), c(1L, n_values))
z_g <- tf$multiply(tf$matmul(x_g, y_g), z_2d)
result7 <- z_g$eval()

# Note: we can list all the operations of a graph
ops <- tf$get_default_graph()$get_operations()
op_names <- NULL
for (op in ops) {
  op_names <- c(op_names, op$name)
}

# Now create a function to compute the Gabor filter
gabor <- function(n_values=32L, sigma=1.0, mean=0.0) {
  x <- tf$linspace(-3.0, 3.0, n_values)
  z <- (tf$exp(tf$negative(tf$pow(x - mean, 2.0) / (2.0 * tf$pow(sigma, 2.0)))) * (1.0 / (sigma * tf$sqrt(2.0 * 3.14159))))
  gauss_kernel <- tf$matmul(tf$reshape(z, c(n_values, 1L)), tf$reshape(z, c(1L, n_values)))
  x <- tf$reshape(tf$sin(tf$linspace(-3.0, 3.0, n_values)), c(n_values, 1L))
  y <- tf$reshape(tf$ones_like(x), c(1L, n_values))
  gabor_kernel <- tf$multiply(tf$matmul(x, y), gauss_kernel)
  return(gabor_kernel)
}

# Verify that the above function does something...
result8 <- gabor()$eval()
comparison_gabor <- result7[32, 32] == result8[32, 32]

# Let's add a function to convolve
convolve <- function(img, W) {
  # The W matrix is 2D
  # But conv2d will need a 4D tensor:
  # height x width x n_input x n_output
  if (length(W$get_shape()) == 2) {
    dims <- c(W$get_shape()$as_list(), 1L, 1L)
    W <- tf$reshape(W, dims)
  }

  cat("Inside convolve 1")

  if (length(img$get_shape()$as_list()) == 2) {
    cat("2D")
    # num x height x width x channels
    dims <- c(1L, img$get_shape()$as_list(), 1L)
    img <- tf$reshape(img, dims)
  } else if (length(img$get_shape()$as_list()) == 3) {
    cat("Find dims")
    dims <- c(1L, img$get_shape()$as_list())
    cat("reshape")
    img <- tf$reshape(img, dims)
    # If the image is 3 channels then our convolution kernel
    # needs to be repeated for each input channel
    cat("turn W to 3D")
    W <- tf$concat(axis=2L, values=c(W, W, W))
  }

  cat("Inside convolve 2")

  # NB: "stride" is how many values to skip for the dimensions
  # of num, height, width, channels
  convolved <- tf$nn$conv2d(img, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
  return(convolved)
}

# Load up an image
img <- tf$image$decode_jpeg(tf$read_file('./pkmital/data/bus.jpg'))

# Create placeholder for graph that can store any input...

# ... and a graph that can convolve the image with a gabor filter

# Finally, send the image into the graph
