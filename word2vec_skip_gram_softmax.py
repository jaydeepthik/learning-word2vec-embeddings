# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:34:02 2019

@author: jaydeep thik
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()


sentences = [ "Deep learning also known as deep structured learning or hierarchical learning is part of a broader family of machine learning methods based on learning data representations as opposed to task-specific algorithms Learning can be supervised semi-supervised or unsupervised",
             "Deep learning architectures such as deep neural networks deep belief networks and recurrent neural networks have been applied to fields including computer vision speech recognition natural language processing audio recognition social network filtering machine translation bioinformatics drug design medical image analysis material inspection and board game programs where they have produced results comparable to and in some cases superior to human experts",
             "Deep learning models are vaguely inspired by information processing and communication patterns in biological nervous systems yet have various differences from the structural and functional properties of biological brains especially human brains which make them incompatible with neuroscience evidences",
              "Most modern deep learning models are based on an artificial neural network although they can also include propositional formulas or latent variables organized layer-wise in deep generative models such as the nodes in deep belief networks and deep Boltzmann machines",
              "In deep learning, each level learns to transform its input data into a slightly more abstract and composite representation. In an image recognition application, the raw input may be a matrix of pixels; the first representational layer may abstract the pixels and encode edges; the second layer may compose and encode arrangements of edges; the third layer may encode a nose and eyes; and the fourth layer may recognize that the image contains a face. Importantly, a deep learning process can learn which features to optimally place in which level on its own",
              "Deep learning architectures are often constructed with a greedy layer-by-layer method", 
              "Deep learning helps to disentangle these abstractions and pick out which features improve performance",
              "For supervised learning tasks, deep learning methods obviate feature engineering, by translating the data into compact intermediate representations akin to principal components, and derive layered structures that remove redundancy in representation",
              "Deep learning algorithms can be applied to unsupervised learning tasks. This is an important benefit because unlabeled data are more abundant than labeled data. Examples of deep structures that can be trained in an unsupervised manner are neural history compressors"
              ]


word_sequence = " ".join(sentences).lower().split()
word_list = word_sequence.copy()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}

batch_size = 20
embedding_dim =2
vocab_size = len(word_list)

def random_batch(data, size):
    batch_input = []
    batch_labels = []
    
    rand_idx = np.random.choice(range(len(data)), size, replace = False)
    
    for i in rand_idx:
        batch_input.append(np.eye(vocab_size)[data[i][0]])
        batch_labels.append(np.eye(vocab_size)[data[i][1]])
    
    return batch_input, batch_labels

skip_grams = [] #target-context pair

for i in range(2, len(word_sequence)-2):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i-2]], word_dict[word_sequence[i-1]],word_dict[word_sequence[i+1]],word_dict[word_sequence[i+2]]]
    
    for w in context:
        skip_grams.append([target, w])

inputs = tf.placeholder(tf.float32, shape=[None, vocab_size])
labels = tf.placeholder(tf.float32, shape=[None, vocab_size])

W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0,1.0))
WT = tf.Variable(tf.random_uniform([embedding_dim, vocab_size], -1.0,1.0))

hidden_layer = tf.matmul(inputs, W)
output_layer = tf.matmul(hidden_layer, WT)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001)

train_step = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for epoch in range(5000):
        batch_input, batch_labels = random_batch(skip_grams, batch_size)
        _ , loss = sess.run([train_step, cost], feed_dict={inputs:batch_input, labels:batch_labels})
        
        if epoch %1000==0:
             print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    trained_embeddings = W.eval()

for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    
    plt.scatter(x, y)
    plt.annotate(label, xy =(x, y) , xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.figure(figsize=(8, 8), dpi=80)    
plt.show()
    
