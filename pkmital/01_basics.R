# Adapted from Python tensorflow tutorials at
# https://github.com/pkmital/tensorflow_tutorials

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
comparison <- z$graph$version == tf$get_default_graph()$version
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
