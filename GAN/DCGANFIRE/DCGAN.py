from __future__ import division
import time
from ops import *
from utils import *
class DCGAN(object):
    model_name = "DCGAN"
    def __init__(self, sess, epoch, batch_size,dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.coord = tf.train.Coordinator()
        self.input_height = 256
        self.input_width = 256
        self.output_height = 256
        self.output_width = 256

        self.z_dim = 100
        self.c_dim = 3
        self.lambd = 10
        self.disc_iters =5
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.sample_num = 64  # number of generated images to be saved
        threads = tf.train.start_queue_runners(sess=self.sess)
        self.data_X = read_and_decode(self.dataset_name)
        self.num_batches = 32
    def discriminator(self, x, is_training=True, reuse=False):

        with tf.variable_scope("discriminator", reuse=reuse):
            print("D:", x.get_shape())  # 32, 32, 3 = 3072
            net = lrelu(bn(conv2d(x, 32, 3, 3, 2, 2, name='d_conv1' + '_' + self.dataset_name),is_training=is_training,scope='d_bn1'))
            print("D:", net.get_shape())
            net = lrelu(bn(conv2d(net, 64, 3, 3, 2, 2, name='d_conv2' + '_' + self.dataset_name),is_training=is_training,scope='d_bn2'))
            print("D:", net.get_shape())
            net = lrelu(bn(conv2d(net, 128, 3, 3, 2, 2, name='d_conv3' + '_' + self.dataset_name),is_training=is_training,scope='d_bn3'))
            print("D:", net.get_shape())
            net = lrelu(bn(conv2d(net, 256, 3, 3, 2, 2, name='d_conv4' + '_' + self.dataset_name),is_training=is_training,scope='d_bn4'))
            print("D:", net.get_shape())
            net = lrelu(
                    bn(conv2d(net, 512, 3, 3, 2, 2, name='d_conv5' + '_' + self.dataset_name),is_training=is_training,scope='d_bn5'))
            print("D:", net.get_shape())
            net = lrelu(
                    bn(conv2d(net, 512, 3, 3, 2, 2, name='d_conv6' + '_' + self.dataset_name),is_training=is_training,scope='d_bn6'))
            print("D:", net.get_shape())
            net = lrelu(
                   bn(conv2d(net, 512, 3, 3, 2, 2, name='d_conv7' + '_' + self.dataset_name),is_training=is_training,scope='d_bn7'))
            print("D:", net.get_shape())
            net = tf.reshape(net, [self.batch_size, -1])
            print("D:", net.get_shape())
            out_logit = linear(net, 1, scope='d_fc8' + '_' + self.dataset_name)
            print("D:", net.get_shape())
            out = tf.nn.sigmoid(out_logit)
            print("D:", out.get_shape())
            print("------------------------")
        return out, out_logit, net

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
           # h_size = 256
            h_size_2 = 128
            h_size_4 = 64
            h_size_8 = 32
            h_size_16 = 16
            h_size_32 = 8
            h_size_64 = 4
            h_size_128 = 2

            print("G:", z.get_shape())
            net = linear(z, 512 * h_size_128 * h_size_128, scope='g_fc1' + '_' + self.dataset_name)
            print("G:", net.get_shape())
            net = tf.nn.relu(
                    bn(tf.reshape(net, [self.batch_size, h_size_128, h_size_128, 512]), is_training=is_training,
                       scope='g_bn1'))
            print("G:", net.get_shape())
            net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_64, h_size_64, 512], 3, 3, 2, 2,
                                name='g_dc2' + '_' + self.dataset_name), is_training=is_training, scope='g_bn2'))
            print("G:", net.get_shape())
            net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_32, h_size_32, 512], 3, 3, 2, 2,
                                name='g_dc3' + '_' + self.dataset_name), is_training=is_training, scope='g_bn3'))
            print("G:", net.get_shape())
            net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_16, h_size_16, 256], 3, 3, 2, 2,
                                name='g_dc4' + '_' + self.dataset_name), is_training=is_training, scope='g_bn4'))
            print("G:", net.get_shape())
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, h_size_8, h_size_8, 128], 3, 3, 2, 2,
                            name='g_dc5' + '_' + self.dataset_name), is_training=is_training, scope='g_bn5'))
            print("G:", net.get_shape())
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, h_size_4, h_size_4, 64], 3, 3, 2, 2,
                            name='g_dc6' + '_' + self.dataset_name), is_training=is_training, scope='g_bn6'))
            print("G:", net.get_shape())
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, h_size_2, h_size_2, 32], 3, 3, 2, 2,
                            name='g_dc7' + '_' + self.dataset_name), is_training=is_training, scope='g_bn7'))
            print("G:", net.get_shape())
            out = tf.nn.sigmoid(
                    deconv2d(net, [self.batch_size, self.output_height, self.output_width, self.c_dim], 3, 3, 2, 2,
                             name='g_dc8' + '_' + self.dataset_name))
            print("G:", out.get_shape())
            print("------------------------")
        return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        """ Loss Function """
        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)


        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake


        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate , beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
        # saver to save model
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            for idx in range(start_batch_id, self.num_batches):
                threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
                batch_images = self.sess.run(self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size])
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                counter += 1
                # display training status


                # save training results for every 300 steps
                if np.mod(counter, 16) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
            start_batch_id = 0
            self.save(self.checkpoint_dir, counter)
            self.visualize_results(epoch)
        self.save(self.checkpoint_dir, counter)


    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = (np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0