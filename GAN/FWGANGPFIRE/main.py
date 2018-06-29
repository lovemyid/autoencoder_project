from WGAN_GP import WGAN_GP
from utils import show_all_variables
from utils import check_folder
import tensorflow as tf

def main():
    with tf.Session() as sess:
        gan = WGAN_GP(sess,
                      epoch=10000,
                      batch_size=16,
                      dataset_name= 'eyes.tfrecords',
                      checkpoint_dir='checkpoint',
                      result_dir='results',
                      log_dir='logs')
        # build graph
        gan.build_model()
        show_all_variables()
        gan.train()

        print(" [*] Training finished!")
        gan.visualize_results(20-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':

    main()