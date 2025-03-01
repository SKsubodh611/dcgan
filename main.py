from imports import *

tf.keras.backend.clear_session()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
print( "------------------",x_train.shape, x_test.shape)


# We plot first 25 images of training dataset
plt.figure(figsize =(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(x_train[i], cmap = plt.cm.binary)
# print(plt.show())


# code
batch_size = 32
# This dataset fills a buffer with buffer_size elements,
# then randomly samples elements from this buffer, 
# replacing the selected elements with new elements.
def create_batch(x_train):
  dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
  # Combines consecutive elements of this dataset into batches.
 
  dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(1)
  # Creates a Dataset that prefetches elements from this dataset
  return dataset

  # code
num_features = 100

generator = keras.models.Sequential([
	keras.layers.Dense(7 * 7 * 128, input_shape =[num_features]),
	keras.layers.Reshape([7, 7, 128]),
	keras.layers.BatchNormalization(),
	keras.layers.Conv2DTranspose(
		64, (5, 5), (2, 2), padding ="same", activation ="selu"),
	keras.layers.BatchNormalization(),
	keras.layers.Conv2DTranspose(
		1, (5, 5), (2, 2), padding ="same", activation ="tanh"),
])
print("GGGGGGGGGGGGGGGGG",generator.summary())
















discriminator = keras.models.Sequential([
	keras.layers.Conv2D(64, (5, 5), (2, 2), padding ="same", input_shape =[28, 28, 1]),
	keras.layers.LeakyReLU(0.2),
	keras.layers.Dropout(0.3),
	keras.layers.Conv2D(128, (5, 5), (2, 2), padding ="same"),
	keras.layers.LeakyReLU(0.2),
	keras.layers.Dropout(0.3),
	keras.layers.Flatten(),
	keras.layers.Dense(1, activation ='sigmoid')
])
print("DDDDDDDDDDDDDDDDDDDDDDDDDDD",discriminator.summary())












# compile discriminator using binary cross entropy loss and adam optimizer
discriminator.compile(loss ="binary_crossentropy", optimizer ="adam")
# make discriminator no-trainable as of now
discriminator.trainable = False
# Combine both generator and discriminator
gan = keras.models.Sequential([generator, discriminator])
# compile generator using binary cross entropy loss and adam optimizer

gan.compile(loss ="binary_crossentropy", optimizer ="adam")
















seed = tf.random.normal(shape =[batch_size, 100])

def train_dcgan(gan, dataset, batch_size, num_features, epochs = 5):
	generator, discriminator = gan.layers
	for epoch in tqdm(range(epochs)):
		print()
		print("Epoch {}/{}".format(epoch + 1, epochs))

		for X_batch in dataset:
			# create a random noise of sizebatch_size * 100
			# to passit into the generator
			noise = tf.random.normal(shape =[batch_size, num_features])
			generated_images = generator(noise)

			# take batch of generated image and real image and
			# use them to train the discriminator
			X_fake_and_real = tf.concat([generated_images, X_batch], axis = 0)
			y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
			discriminator.trainable = True
			discriminator.train_on_batch(X_fake_and_real, y1)

			# Here we will be training our GAN model, in this step
			# we pass noise that uses generatortogenerate the image
			# and pass it with labels as [1] So, it can fool the discriminator
			noise = tf.random.normal(shape =[batch_size, num_features])
			y2 = tf.constant([[1.]] * batch_size)
			discriminator.trainable = False
			gan.train_on_batch(noise, y2)

			# generate images for the GIF as we go
			generate_and_save_images(generator, epoch + 1, seed)

	generate_and_save_images(generator, epochs, seed)













    #generate_and_save_images function will generate the images and save them
# code
# def generate_and_save_images(model, epoch, test_input):
#   predictions = model(test_input, training = False)
 
#   fig = plt.figure(figsize =(10, 10))
 
#   for i in range(25):
#       plt.subplot(5, 5, i + 1)
#       plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap ='binary')
#       plt.axis('off')
 
#   plt.savefig('image_epoch_{:04d}.png'.format(epoch))








def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
        plt.axis('off')

    plt.savefig(f'image_epoch_{epoch:04d}.png')
    plt.close(fig)  








  # crearing batches of datasets for training the model
  # reshape to add a color map
x_train_dcgan = x_train.reshape(-1, 28, 28, 1) * 2. - 1.
# create batches
dataset = create_batch(x_train_dcgan)
# callthe training function with 10 epochs and record time %% time
train_dcgan(gan, dataset, batch_size, num_features, epochs = 10)










 
anim_file = 'dcgan_results.gif'
 
with imageio.get_writer(anim_file, mode ='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i, filename in enumerate(filenames):
    frame = 2*(i)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
display.Image(filename = anim_file)


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


embed.embed_file(anim_file)
