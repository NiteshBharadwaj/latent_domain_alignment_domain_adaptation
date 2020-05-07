import tensorflow as tf
import time

if __name__ == '__main__':

	with tf.Graph().as_default():

		A = tf.zeros(dtype=tf.float32, shape=[1, 2048, 2048])
		B = tf.zeros(dtype=tf.float32, shape=[1, 2048, 2048])
	
		C = A + B

		with tf.Session() as sess:

			while True:
				sess.run(C)
				time.sleep(0.005)	
			
			

