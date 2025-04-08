import jax
import jax.numpy as jnp

def convert_L(image):
  #convert image to greyscale using the ITU-R 601-2 luma transform
  return jnp.maximum(jnp.minimum(image[:,:,0] * 0.299 + image[:,:,1] * 0.587 + image[:,:,2] * 0.114, 255), 0).astype("uint8")

def phash_jax(image, hash_size=8, highfreq_factor=4):
  img_size = hash_size * highfreq_factor
  image = jax.image.resize(convert_L(image), [img_size, img_size], "lanczos3") #convert to greyscale
  dct = jax.scipy.fft.dct(jax.scipy.fft.dct(image, axis=0), axis=1)
  dctlowfreq = dct[:hash_size, :hash_size]
  med = jnp.median(dctlowfreq)
  diff = dctlowfreq > med
  return diff

def hash_dist(h1, h2):
  return jnp.count_nonzero(h1.flatten() != h2.flatten())

batch_phash = jax.vmap(jax.jit(phash_jax))

