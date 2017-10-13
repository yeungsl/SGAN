import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import scipy
import scipy.misc
from numpy import linalg as la

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

# ===== Encoder Start ===== #

def dnn(x, Train=True): # Encoder
    with tf.variable_scope("encoder") as scope:
        if Train is False:
            scope.reuse_variables()

        '''input layer'''

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        encoders = [x_image]

        ''' convolution layer 1'''

        W_conv1 = tf.get_variable("W_conv1", shape=[5,5,1,32], initializer=tf.contrib.layers.xavier_initializer()) # define filters
        b_conv1 = tf.get_variable("b_conv1", shape=[32], initializer=tf.constant_initializer(0.1))
        l_conv1 = tf.nn.relu(tf.nn.conv2d(encoders[-1], W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
        encoders.append(l_conv1)

        ''' pooling layer 1 '''

        l_pool1 = tf.nn.max_pool(encoders[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        encoders.append(l_pool1)

        ''' convolution layer 2'''

        W_conv2 = tf.get_variable("W_conv2", shape=[5,5,32,64], initializer=tf.contrib.layers.xavier_initializer())
        b_conv2 = tf.get_variable("b_conv2", shape=[64], initializer=tf.constant_initializer(0.1))
        l_conv2 = tf.nn.relu(tf.nn.conv2d(encoders[-1], W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
        encoders.append(l_conv2)

        ''' pooling layer 2'''

        l_pool2 = tf.nn.max_pool(encoders[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        encoders.append(l_pool2)

        ''' fully connected layer'''
        W_fc1 = tf.get_variable("W_fc1", shape=[7*7*64, 256], initializer=tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.get_variable("b_fc1", shape=[256], initializer=tf.constant_initializer(0.1))

        l_pool2_flat = tf.reshape(encoders[-1], [-1, 7 * 7 * 64])
        encoders.append(l_pool2_flat)
        l_fc1 = tf.nn.relu(tf.matmul(encoders[-1], W_fc1) + b_fc1)
        encoders.append(l_fc1)

        ''' Dropout layer (DID NOT appear in SGAN's implementation)'''
        keep_prob = tf.placeholder(tf.float32)
        l_fc1_drop = tf.nn.dropout(encoders[-1], keep_prob)
        encoders.append(l_fc1_drop)

        ''' Map the features to 10 attprint(refy_1hot.shape)ribute (output layer)'''
        W_fc2 = tf.get_variable("W_fc2", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.get_variable("b_fc2", shape=[10], initializer=tf.constant_initializer(0.1))

        l_fc2 = tf.matmul(encoders[-1], W_fc2) + b_fc2
        encoders.append(l_fc2)

        if Train is False:
            return encoders, keep_prob, [W_fc2, b_fc2]
        else:
            return encoders, keep_prob

def loss_encoder(encoders, y):
    # y --- input label
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=encoders[-1]))

def opt_encoder(loss_func, lr, n):
    # adam optimizor
    # loss_func: using learning_rate = 1e-4
    # TODO: Learning Rate for training
    return tf.train.AdamOptimizer(learning_rate=lr, name=n).minimize(loss_func)

def acc_encoder(y_conv, y):
    # y_conv: encoder output
    # y: input label (same as y)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    """ p.s.  tf.argmax(input, axis=None) Returns the index with the largest value across axes of a tensor. """
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(correct_prediction)
# ===== Encoder END ===== #


# ====== Generator Begin ======== #
def generator_1(y_hot, batch_size, Train=True):
    # z1 noise
    # y_hot
    with tf.variable_scope('generator_1') as scope:
        if Train is False:
            scope.reuse_variables()

        z1 = tf.get_variable('z1', [batch_size, 50], initializer=tf.random_uniform_initializer(maxval=1.0))
        generator_1 = [z1]
        l_input = tf.reshape(y_hot, [batch_size, 10]) # the same with -1 in enc                L1_norm_samp_ori = la.norm(imgs - orix, 1)oder batch_size
        generator_1.append(l_input)
        generator_1.append(tf.concat([z1, l_input], 1)) # along row as 1 ---> output shape = batch_size * 60
        generator_1.append(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_1[-1], 512)))
        generator_1.append(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_1[-1], 512)))
        generator_1.append(tf.contrib.layers.fully_connected(generator_1[-1], 256)) # -> output shape = [batch_size, 256]

        return generator_1, z1

def generator_0(fc3, batch_size, Train=True):
    # z0  --- another noise
    # fc3 ---
    #   Option1: generator_1's output with shape = [batch_size, 256]
    #   Option2: Encoder's Output with encoder[-3], with output shae = [batch_size, 256]
    with tf.variable_scope('generator_0') as scope:
        if Train is False:
            scope.reuse_variables()
        z0 = tf.get_variable('z0', [batch_size, 50], initializer=tf.random_uniform_initializer(maxval=1.0))
        generator_0 = [z0]
        generator_0.append(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_0[-1], 128)))
        # embed noise Z into generator
        gen0_z_embed = generator_0[-1]
        # generator_1's (or Encoder's) output into generator
        generator_0.append(tf.reshape(fc3, [batch_size, 256]))

        # generator_1's output + noise_Z0
        generator_0.append(tf.concat([generator_0[-1], gen0_z_embed], 1))

        generator_0.append(tf.reshape(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_0[-1], 128*4*4)),[batch_size, 4, 4, 128]))

        generator_0.append(tf.pad(tf.contrib.layers.batch_norm(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(generator_0[-1],
                                                                                                                tf.get_variable('gen0_conv2d0_W', [5,5,128,128], initializer=tf.random_normal_initializer(stddev=0.02)),
                                                                                                                [batch_size,8,8,128],
                                                                                                                [1,2,2,1]),
                                                                                         tf.get_variable('gen0_conv2d0_b', [128], initializer=tf.constant_initializer(0.0))))),
                                  [[0,0], [2,2], [2,2], [0,0]],
                                  "CONSTANT"))

        generator_0.append(tf.contrib.layers.batch_norm(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(generator_0[-1],
                                                                                                         tf.get_variable('gen0_conv2d1_W', [5,5,64,128], initializer=tf.random_normal_initializer(stddev=0.02)),
                                                                                                         [batch_size,12,12,64],
                                                                                                         [1,1,1,1]),
                                                                                  tf.get_variable('gen0_conv2d1_b', [64], initializer=tf.constant_initializer(0.0))))))

        generator_0.append(tf.pad(tf.contrib.layers.batch_norm(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(generator_0[-1],
                                                                                                                tf.get_variable('gen0_conv2d2_W', [5,5,64,64], initializer=tf.random_normal_initializer(stddev=0.02)),
                                                                                                                [batch_size,24,24,64],
                                                                                                                [1,2,2,1]),
                                                                                         tf.get_variable('gen0_conv2d2_b', [64], initializer=tf.constant_initializer(0.0))))),
                                  [[0,0], [2,2], [2,2], [0,0]],
                                  "CONSTANT"))

        generator_0.append(tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d_transpose(generator_0[-1],
                                                                               tf.get_variable('gen0_conv2d3_W', [5,5,1,64], initializer=tf.random_normal_initializer(stddev=0.02)),
                                                                               [batch_size,28,28,1],
                                                                               [1,1,1,1]),
                                                        tf.get_variable('gen0_conv2d3_b', [1], initializer=tf.constant_initializer(0.0)))))

        return generator_0, z0

def discriminator_1(in_l):
    # in_l: Encoder Output / Generator_1 Output
    discriminator_1 = [in_l]
    # TODO:WHY Two Layer
    discriminator_1.append(tf.contrib.layers.fully_connected(discriminator_1[-1], 256, activation_fn=lrelu))
    discriminator_1.append(tf.contrib.layers.fully_connected(discriminator_1[-1], 256, activation_fn=lrelu))
    disc_1_shared = discriminator_1[-1]
    discriminator_1.append(tf.contrib.layers.fully_connected(disc_1_shared, 50, activation_fn=tf.sigmoid))
    discriminator_1.append(tf.contrib.layers.fully_connected(disc_1_shared, 1, activation_fn=tf.sigmoid))
    return discriminator_1

def discriminator_0(in_l, batch_size):
    # in_l: input layer from Real/Fake Image
    discriminator_0 = [in_l]
    discriminator_0.append(tf.reshape(discriminator_0[-1], [batch_size, 28, 28, 1]))
    discriminator_0.append(tf.contrib.layers.conv2d(discriminator_0[-1], 32, [5,5], stride=2, activation_fn=lrelu))
    discriminator_0.append(tf.contrib.layers.conv2d(discriminator_0[-1], 64, [5,5], stride=2, activation_fn=lrelu))
    discriminator_0.append(tf.contrib.layers.conv2d(discriminator_0[-1], 128, [5,5], stride=2, activation_fn=lrelu))
    discriminator_0.append(tf.contrib.layers.fully_connected(discriminator_0[-1], 256, activation_fn=lrelu))
    disc_0_shared = discriminator_0[-1] # the output of 256
    discriminator_0.append(tf.contrib.layers.fully_connected(disc_0_shared, 50, activation_fn=tf.sigmoid))
    # If input is  Real or Fake Image
    discriminator_0.append(tf.contrib.layers.fully_connected(disc_0_shared, 1, activation_fn=tf.sigmoid))
    return discriminator_0

if __name__ == '__main__':

    ''' input variables '''
    batch_size = 100
    learning_rate = 0.000001
    Epoch = 10000
    out_dir = os.getcwd()+"/output_img"
    # from paper
    condloss_weight = 1.0
    advloss_weight = 1.0
    entloss_weight = 10.0
    # from paper end

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    ''' build encoder input and encoder layers'''
    x = tf.placeholder(tf.float32, [None, 784]) # x input image with shape=(28*28).flatten
    y = tf.placeholder(tf.float32, [None, 10]) # y label with shape=10
    # Feed Encoder

    pre_trained_encoders, keep_prob = dnn(x)
    # used for DEBUG to check the encoder's last output
    print('encoder fully connected last layer with label_shape = 10:', pre_trained_encoders[-1])
    # DEBUG END

    ''' training steps for encoders'''
    cross_entropy = loss_encoder(pre_trained_encoders, y) # Calculate Cross_Entropy
    train_steps = opt_encoder(cross_entropy, learning_rate, 'encoder_only')  # Feed Cross Entropy to Optimizor, and now it is ready for training :)
    accuracy = acc_encoder(pre_trained_encoders[-1],y)    # Calculate Accuracy

    ''' train the encoders'''


    # ==== training Encoder first ! === #
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(Epoch): # training Step = 20000 No Epoch ...
            batch = mnist.train.next_batch(50)

            # ---  For optimizor --- #
            train_steps.run(feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})

            # ---  For Train Accuracy --- #
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
                print('step %d, training accuracy %f' %(i, train_accuracy))


    encoders, _, reconstruction_var = dnn(x, Train=False)
    real_fc3 = encoders[-3] # <- extract output of Fully_Connected_1 with shape=[batch, 256]
    print('encoder real ???', real_fc3)

    ''' generater 1 '''
    y_hot = tf.placeholder(tf.float32, [None, None])
    gen_fc3, z1 = generator_1(y_hot, batch_size)
    print('generator 1 last layer 256:', gen_fc3[-1])

    ''' generator 0 '''
    gen_x, z0 = generator_0(real_fc3, batch_size)
    gen_x_joint, _ = generator_0(gen_fc3[-1], batch_size, Train=False)
    print('generator0 for encoder1:', gen_x[-1])
    print('generator0 for generator1:', gen_x_joint[-1])


    ''' forward pass '''
    # discriminator1
    dis_real1 = discriminator_1(real_fc3)
    prob_real1 = dis_real1[-1]
    print('discriminator1 for encoders 1:', prob_real1)
    dis_gen1 = discriminator_1(gen_fc3[-1])
    prob_gen1 = dis_gen1[-1]
    recon_z1 = dis_gen1[-2]
    print('discriminator1 for generator1 1:', prob_gen1)
    print('discriminator1 for reconize z 50:', recon_z1)

    # discriminator0
    dis_real0 = discriminator_0(x, batch_size)
    prob_real0 = dis_real0[-1]
    print('discriminator0 for input:', prob_real0)
    dis_gen0 = discriminator_0(gen_x[-1], batch_size)
    prob_gen0 = dis_gen0[-1]
    recon_z0 = dis_gen0[-2]
    print('discriminator0 for generator0 1:', prob_gen0)
    print('discriminator0 for reconize z 50:', recon_z0)


    ''' loss function '''
    # discriminator1
    loss_real1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_real1, tf.ones(tf.shape(prob_real1))))
    loss_fake1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen1, tf.zeros(tf.shape(prob_gen1))))
    loss_gen1_ent = tf.reduce_mean((recon_z1 - z1) ** 2)
    # from paper
    loss_dis1 = advloss_weight * (0.5*loss_real1 + 0.5*loss_fake1) + entloss_weight * loss_gen1_ent

    # discriminator0
    loss_real0 = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_real0, tf.ones(tf.shape(prob_real0))))
    loss_fake0 = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen0, tf.zeros(tf.shape(prob_gen0))))
    loss_gen0_ent = tf.reduce_mean((recon_z0 - tf.pad(tf.expand_dims(tf.expand_dims(z0, 1),1), [[0,0],[1,2],[1,2],[0,0]],"CONSTANT") ** 2))
    loss_dis0 = advloss_weight * (0.5*loss_real0 + 0.5*loss_fake0) + entloss_weight * loss_gen0_ent

    # generator1
    recon_y = tf.matmul(gen_fc3[-1], reconstruction_var[0]) + reconstruction_var[1]
    loss_gen1_adv = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen1, tf.ones(tf.shape(prob_gen1))))
    loss_gen1_cond = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=recon_y, logits=y_hot))
    loss_gen1 = advloss_weight * loss_gen1_adv + condloss_weight * loss_gen1_cond + entloss_weight * loss_gen1_ent

    # generator0
    recon_fc3, _, _ = dnn(gen_x[-1], Train=False)
    loss_gen0_adv = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen0, tf.ones(tf.shape(prob_gen0))))
    loss_gen0_cond = tf.reduce_mean((recon_fc3[-3] - real_fc3)**2)
    loss_gen0 = advloss_weight * loss_gen0_adv + condloss_weight * loss_gen0_cond + entloss_weight * loss_gen0_ent

    ''' training steps '''
    train_dis1 = opt_encoder(loss_dis1, learning_rate, 'pro_encoder')
    train_dis0 = opt_encoder(loss_dis0, learning_rate, 'pro_encoder')
    train_gen1 = opt_encoder(loss_gen1, learning_rate, 'pro_encoder')
    train_gen0 = opt_encoder(loss_gen0, learning_rate, 'pro_encoder')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        for i in range(Epoch):
            batch = mnist.train.next_batch(batch_size)
            batch_hot = np.zeros((batch_size, 10), dtype=np.float32)
            for j in range(len(batch[1])):
                batch_hot[j][np.argmax(batch[1][j])] = 1.

            sess.run(train_gen1, feed_dict = {y_hot:batch_hot})
            sess.run(train_gen0, feed_dict = {x: batch[0]})
            sess.run(train_dis1, feed_dict = {x: batch[0], y_hot:batch_hot})
            sess.run(train_dis0, feed_dict = {x: batch[0]})

            if i % 1000 == 0:
                gen0_l = loss_gen0.eval(feed_dict={x:batch[0]})
                gen1_l = loss_gen1.eval(feed_dict={y_hot:batch_hot})
                dis1_l = loss_dis1.eval(feed_dict={x:batch[0], y_hot:batch_hot})
                dis0_l = loss_dis0.eval(feed_dict={x:batch[0]})
                print("Step %d, gen1 loss: %f, gen0 loss: %f, dis1 loss: %f, dis0 loss: %f"%(i, gen1_l, gen0_l, dis1_l, dis0_l))


                ## reconstruct image from stached generator ##

                refy = np.zeros((batch_size), dtype=np.int)
                for l in range(batch_size):
                    refy[l] = l%10
                refy_1hot = np.zeros((batch_size, 10), dtype=np.float32)
                refy_1hot[np.arange(batch_size), refy] = 1

                imgs = sess.run(gen_x_joint[-1],feed_dict={y_hot:refy_1hot})

                imgs = np.reshape(imgs[:100,], (100, 28, 28))
                '''
                imgs = [imgs[i, :, :] for i in range (100)]
                rows_gen = []
                for k in range(10):
                    rows_gen.append(np.concatenate(imgs[k::10], 1))
                imgs = np.concatenate(rows_gen, 0)
                scipy.misc.imsave(out_dir+"/mnist_sample_epoch{}.png".format(i), imgs)
                '''
                ## generate original image ##
                orix = np.reshape(batch[0][:100, ], (100, 28, 28))
                L1_norm_samp_ori = np.mean(imgs - orix)
                L2_norm_samp_ori = la.norm(imgs - orix)
                '''
                orix = [orix[i, :, :] for i in range(100)]
                rows_ori = []
                for k in range(10):
                    rows_ori.append(np.concatenate(orix[k::10], 1))
                orix = np.concatenate(rows_ori, 0)
                scipy.misc.imsave(out_dir+"/mnist_ori_epoch{}.png".format(i), imgs)
                '''
                ## reconstruct image from encoders ##

                reconx = sess.run(gen_x[-1], feed_dict={x:batch[0]})
                reconx = np.reshape(reconx[:100], (100, 28, 28))
                L1_norm_r_ori = np.mean(reconx - orix)
                L2_norm_r_ori = la.norm(reconx - orix)
                '''
                reconx = [reconx[i, :, :] for i in range(100)]
                rows_r = []
                for k in range(10):
                    rows_r.append(np.concatenate(reconx[k::10], 1))
                reconx = np.concatenate(rows_r, 0)
                scipy.misc.imsave(out_dir+"/mnist_recon_epoch{}.png".format(i), imgs)
                '''

                print("Step %d, L1_sample_ori: %f, L2_sample_ori: %f, L1_recon_ori: %f, L2_recon_ori: %f"%(i, L1_norm_samp_ori, L2_norm_samp_ori, L1_norm_r_ori, L2_norm_r_ori))


