# From https://rstudio.github.io/tensorflow/

library(tensorflow)

# Create 100 phony x, y, z data points, z = x * 0.1 + y * 0.2 + 0.3
x_data <- runif(100, min=0, max=1)
y_data <- runif(100, min=0, max=1)
z_data <- x_data * 0.1 + y_data * 0.2 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
A <- tf$Variable(tf$random_uniform(shape(1L), -1.0, 1.0))
B <- tf$Variable(tf$random_uniform(shape(1L), -1.0, 1.0))
C <- tf$Variable(tf$zeros(shape(1L)))
z <- A * x_data + B * y_data + C

# Minimize the mean squared errors.
loss <- tf$reduce_mean((z - z_data) ^ 2)
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train <- optimizer$minimize(loss)

# Launch the graph and initialize the variables.
sess = tf$Session()
sess$run(tf$global_variables_initializer())

# Fit the line (Learns best fit is W: 0.1, b: 0.3)
for (step in 1:201) {
  sess$run(train)
  if (step %% 20 == 0)
    cat(step, "-", sess$run(A), sess$run(B), sess$run(C), "\n")
}
