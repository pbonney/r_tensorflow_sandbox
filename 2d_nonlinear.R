library(tensorflow)
library(ggplot2)

x_data <- runif(1000, min=-1, max=1)
y_data <- runif(1000, min=-1, max=1)
z_data <- as.matrix(2 * x_data^2 - 3 * y_data^2 + 1)

sess <- tf$InteractiveSession()

print("building model")
L <- tf$placeholder(tf$float32, shape(NULL, 2L), name='L')

D_h <- 10L

W_in <- tf$Variable(tf$truncated_normal(shape(2L, D_h), stddev=0.1), name='W_in')
b_in <- tf$Variable(tf$zeros(shape(1L, D_h)), name='b_in')
M <- tf$tanh(tf$matmul(L, W_in) + b_in)

W_out <- tf$Variable(tf$truncated_normal(shape(D_h, 1L), stddev=0.1), name='W_out')
b_out <- tf$Variable(tf$zeros(shape(1L, 1L)), name='b_out')
z <- tf$matmul(M, W_out) + b_out

z_ <- tf$placeholder(tf$float32, shape(NULL, 1L), name='z_')

print("defining training parameters")
loss <- tf$nn$l2_loss(z - z_)
optimizer <- tf$train$AdamOptimizer(5e-2)
train <- optimizer$minimize(loss)

L_in <- cbind(x_data, y_data)
colnames(L_in) <- NULL

sess$run(tf$global_variables_initializer())

print("Train!!!")
for (i in 1:1000) {
  sess$run(train,
           feed_dict = dict(L = L_in, z_ = z_data))
  if (i %% 20 == 0) {
		train_loss <- loss$eval(feed_dict = dict(L = L_in, z_ = z_data))
    cat(sprintf("step %d, training loss %g\n", i, train_loss))
	}
}

x_plot <- seq(-1,1,by=0.025)
y_plot <- seq(-1,1,by=0.025)

grid <- expand.grid(x=x_plot,y=y_plot)
m.grid <- as.matrix(cbind(grid$x, grid$y))
rownames(m.grid) <- NULL

z_tf <- z$eval(feed_dict = dict(L = m.grid))
z_act <- as.matrix(2 * grid$x^2 - 3 * grid$y^2 + 1)

my_theme <- theme(text = element_text(family='Lato'),
                 panel.background = element_rect(fill = "white", color='#BFBFBF'),
                 plot.background = element_rect(fill = "transparent",colour = NA),
                 panel.grid.minor = element_blank(),
                 panel.grid.major = element_blank(),
                 panel.border = element_rect(colour = "black", fill=NA, size=0.5))

plot_function <- function(x, y, z, filename="test.png") {
  coords <- data.frame(x=x,y=y,z=z)
  g <- ggplot(coords,aes(x,y))
  g <- g + geom_tile(aes(fill=z)) + xlab("X") + ylab("Z")
  g <- g + scale_fill_gradient(low="white",high="black")
  g <- g + my_theme
  ggsave(g,file=paste("./",filename,sep=""),height=4,width=5)
	return(TRUE)
}
