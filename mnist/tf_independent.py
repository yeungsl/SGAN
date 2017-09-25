import tensorflow as tf
import numpy as np


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def dnn(x):

    '''input layer'''

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    encoders = [x_image]

    ''' convolution layer 1'''

    W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    l_conv1 = tf.nn.relu(tf.nn.conv2d(encoders[-1], W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
    encoders.append(l_conv1)

    ''' pooling layer 1 '''

    l_pool1 = tf.nn.max_pool(encoders[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    encoders.append(l_pool1)

    ''' convolution layer 2'''

    W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    l_conv2 = tf.nn.relu(tf.nn.conv2d(encoders[-1], W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    encoders.append(l_conv2)

    ''' pooling layer 2'''

    l_pool2 = tf.nn.max_pool(encoders[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    encoders.append(l_pool2)

    ''' fully connected layer'''
    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 256], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]))

    l_pool2_flat = tf.reshape(encoders[-1], [-1, 7 * 7 * 64])
    encoders.append(l_pool2_flat)
    l_fc1 = tf.nn.relu(tf.matmul(encoders[-1], W_fc1) + b_fc1)
    encoders.append(l_fc1)

    ''' Dropout layer (DID NOT appear in SGAN's implementation)'''
    keep_prob = tf.placeholder(tf.float32)
    l_fc1_drop = tf.nn.dropout(encoders[-1], keep_prob)
    encoders.append(l_fc1_drop)

    ''' Map the features to 10 attribute (output layer)'''
    W_fc2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

    l_fc2 = tf.matmul(encoders[-1], W_fc2) + b_fc2
    encoders.append(l_fc2)

    return encoders, keep_prob, [W_fc1, b_fc1, W_fc2, b_fc2]

def reconstruct_encoder(x, reconstruction_var):
    new_enc = [tf.reshape(x, [-1, 7*7*64])]
    new_enc.append(tf.nn.relu(tf.matmul(new_enc[-1], reconstruction_var[0]) + reconstruction_var[1]))
    new_enc.append(tf.matmul(new_enc[-1], reconstruction_var[2]) + reconstruction_var[3])
    return new_enc

def loss_encoder(encoders, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=encoders[-1]))

def opt_encoder(loss_func):
    return tf.train.AdamOptimizer(1e-4).minimize(loss_func)

def acc_encoder(y_conv, y_):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(correct_prediction)

def generator_1(z1, y_hat, batch_size):

    generator_1 = [z1]
    l_input = tf.reshape(y_hat, [batch_size, 10])
    generator_1.append(l_input)
    generator_1.append(tf.concat([z1, l_input], 1))
    generator_1.append(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_1[-1], 512)))
    generator_1.append(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_1[-1], 512)))
    generator_1.append(tf.contrib.layers.fully_connected(generator_1[-1], 256))

    return generator_1

def generator_0(z0, fc3, batch_size):

    generator_0 = [z0]
    generator_0.append(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_0[-1], 128)))
    gen0_z_embed = generator_0[-1]
    generator_0.append(tf.reshape(fc3, [batch_size, 256]))
    generator_0.append(tf.concat([generator_0[-1], gen0_z_embed], 1))
    generator_0.append(tf.reshape(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(generator_0[-1], 128*4*4)), [batch_size, 4, 4, 128]))
    generator_0.append(tf.contrib.layers.batch_norm(tf.contrib.layers.conv2d_transpose(generator_0[-1], 128, [5,5], stride=2)))
    generator_0.append(tf.contrib.layers.batch_norm(tf.contrib.layers.conv2d_transpose(generator_0[-1], 64, [5,5], padding='VALID')))
    generator_0.append(tf.contrib.layers.batch_norm(tf.contrib.layers.conv2d_transpose(generator_0[-1], 64, [5,5], stride=2, padding='SAME')))
    generator_0.append(tf.contrib.layers.conv2d_transpose(generator_0[-1], 1, [5,5], padding='VALID', activation_fn=tf.sigmoid))

    return generator_0

def discriminator_1(in_l):
    discriminator_1 = [in_l]
    discriminator_1.append(tf.contrib.layers.fully_connected(discriminator_1[-1], 256, activation_fn=lrelu))
    discriminator_1.append(tf.contrib.layers.fully_connected(discriminator_1[-1], 256, activation_fn=lrelu))
    disc_1_shared = discriminator_1[-1]
    discriminator_1.append(tf.contrib.layers.fully_connected(disc_1_shared, 50, activation_fn=tf.sigmoid))
    discriminator_1.append(tf.contrib.layers.fully_connected(disc_1_shared, 1, activation_fn=tf.sigmoid))
    return discriminator_1

def discriminator_0(in_l, batch_size):
    discriminator_0 = [in_l]
    discriminator_0.append(tf.reshape(discriminator_0[-1], [batch_size, 28, 28, 1]))
    discriminator_0.append(tf.contrib.layers.conv2d(discriminator_0[-1], 32, [5,5], stride=2, activation_fn=lrelu))
    discriminator_0.append(tf.contrib.layers.conv2d(discriminator_0[-1], 64, [5,5], stride=2, activation_fn=lrelu))
    discriminator_0.append(tf.contrib.layers.conv2d(discriminator_0[-1], 128, [5,5], stride=2, activation_fn=lrelu))
    discriminator_0.append(tf.contrib.layers.fully_connected(discriminator_0[-1], 256, activation_fn=lrelu))
    disc_0_shared = discriminator_0[-1]
    discriminator_0.append(tf.contrib.layers.fully_connected(disc_0_shared, 50, activation_fn=tf.sigmoid))
    discriminator_0.append(tf.contrib.layers.fully_connected(disc_0_shared, 1, activation_fn=tf.sigmoid))
    return discriminator_0


if __name__ == '__main__':

    ''' input variables '''
    batch_size = 100
    condloss_weight = 1.0
    advloss_weight = 1.0
    entloss_weight = 10.0

    ''' build encoder input and encoder layers'''
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    encoders, keep_prob, reconstruction_var = dnn(x)
    print('encoder fully connected last layer 10:', encoders[-1])

    ''' training steps for encoders'''
    cross_entropy = loss_encoder(encoders, y)
    train_steps = opt_encoder(cross_entropy)
    accuracy = acc_encoder(encoders[-1],y)

    ''' train the encoders'''
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
                print('step %d, training accuraryc %g' %(i, train_accuracy))
            train_steps.run(feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
    '''

    x_g = tf.placeholder(tf.float32, [None, None, None, None])
    real_fc3 = reconstruct_encoder(x_g, reconstruction_var)[-2]
    print('encoder real ???', real_fc3)

    ''' generater 1 '''
    y_hot = tf.placeholder(tf.float32, [None, None])
    z1 = tf.Variable(tf.random_uniform([batch_size, 50], maxval=1.0))
    gen_fc3 = generator_1(z1, y_hot, batch_size)
    print('generator 1 last layer 256:', gen_fc3[-1])

    ''' generator 0 '''
    z0 = tf.Variable(tf.random_uniform([batch_size, 50], maxval=1.0))
    gen_x = generator_0(z0, real_fc3, batch_size)
    print('generator0 for encoder1:', gen_x[-1])


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
    dis_real0 = discriminator_0(x_g, batch_size)
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
    loss_dis1 = advloss_weight * (0.5*loss_real1 + 0.5*loss_fake1) + entloss_weight * loss_gen1_ent

    # discriminator0
    loss_real0 = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_real0, tf.ones(tf.shape(prob_real0))))
    loss_fake0 = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen0, tf.zeros(tf.shape(prob_gen0))))
    loss_gen0_ent = tf.reduce_mean((recon_z0 - tf.reshape(z0, [batch_size, -1, -1, 50]) ** 2))
    loss_dis0 = advloss_weight * (0.5*loss_real0 + 0.5*loss_fake0) + entloss_weight * loss_gen0_ent

    # generator1
    recon_y = tf.matmul(gen_fc3[-1], reconstruction_var[2]) + reconstruction_var[3]
    loss_gen1_adv = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen1, tf.ones(tf.shape(prob_gen1))))
    loss_gen1_cond = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=recon_y, logits=y_hot))
    loss_gen1 = advloss_weight * loss_gen1_adv + condloss_weight * loss_gen1_cond + entloss_weight * loss_gen1_ent

    # generator0
    recon_fc3 = reconstruct_encoder(gen_x[-1], reconstruction_var)[-2]
    loss_gen0_adv = tf.reduce_mean(tf.losses.softmax_cross_entropy(prob_gen0, tf.ones(tf.shape(prob_gen0))))
    loss_gen0_cond = tf.reduce_mean((recon_fc3 - real_fc3)**2)
    loss_gen0 = advloss_weight * loss_gen0_adv + condloss_weight * loss_gen0_cond + entloss_weight * loss_gen0_ent
