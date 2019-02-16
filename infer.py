#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
import os
from tqdm import tqdm
import pickle


def load_vocab(path):
    """
    加载词典
    """
    with open(path, 'r', encoding='utf-8') as fr:
        vocab = fr.readlines()
        vocab = [w.strip('\n') for w in vocab]  # 去掉每行的换行符
    return vocab

vocab_ch = load_vocab('data/vocab.ch')  # 中文词典
vocab_en = load_vocab('data/vocab.en')  # 英文词典

print('=' * 100)
print('中文词典长度：', len(vocab_ch), '中文词典 20 词：', vocab_ch[:20])
print('英文词典长度：', len(vocab_en), '英文词典 20 词：', vocab_en[:20])
print()

word2id_ch = {w: i for i, w in enumerate(vocab_ch)}
id2word_ch = {i: w for i, w in enumerate(vocab_ch)}
word2id_en = {w: i for i, w in enumerate(vocab_en)}
id2word_en = {i: w for i, w in enumerate(vocab_en)}


def load_data(path, word2id):
    """
    加载数据
    """
    with open(path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        sentences = [line.strip('\n').split(' ') for line in lines]
        # 每个句子以 <s> 开头，以 </s> 结束，并且把所有的词转为 id
        sentences = [[word2id['<s>']] + [word2id[w] for w in sentence] + [word2id['</s>']] for sentence in sentences]
        # 统计每个句子的长度
        lens = [len(sentence) for sentence in sentences]
        # 句子最大长度
        maxlen = np.max(lens)
        return sentences, lens, maxlen

# 调用函数，加载数据，统计每个句子的长度，统计句子最大长度
train_ch, len_train_ch, maxlen_train_ch = load_data('data/train.ch', word2id_ch)
train_en, len_train_en, maxlen_train_en = load_data('data/train.en', word2id_en)
dev_ch, len_dev_ch, maxlen_dev_ch = load_data('data/dev.ch', word2id_ch)
dev_en, len_dev_en, maxlen_dev_en = load_data('data/dev.en', word2id_en)
test_ch, len_test_ch, maxlen_test_ch = load_data('data/test.ch', word2id_ch)
test_en, len_test_en, maxlen_test_en = load_data('data/test.en', word2id_en)
# 训练集、验证集、测试集的句子最大长度
maxlen_ch = np.max([maxlen_train_ch, maxlen_dev_ch, maxlen_test_ch])
# 训练集、验证集、测试集的句子最大长度
maxlen_en = np.max([maxlen_train_en, maxlen_dev_en, maxlen_test_en])
# 打印最大句子长度
print('=' * 100)
print('maxlen_ch:', maxlen_ch)
print('maxlen_en:', maxlen_en)
# 查看句子，句子中的词已经转为 id，并且已经添加了 <s> 和 </s>
for i, x in enumerate(train_ch[0:5]):
    print('train_ch[{}]:'.format(i), x)
for i, x in enumerate(train_en[0:5]):
    print('train_en[{}]:'.format(i), x)
print()
# 查看训练集前 20 个样例的句子长度
print('='*100)
print('len_train_ch[0:10]:', len_train_ch[0:10])
print('len_train_en[0:10]:', len_train_en[0:10])
print()


"""
三种模式：

1. train: training, no beam search, calculate loss
2. eval: no training, no beam search, calculate loss
3. infer: no training, beam search, calculate bleu
"""
mode = 'infer'


# 填充
if mode == 'train':
    # 在句子结尾填充，填充值为 </s> 对应的 id
    train_ch = pad_sequences(train_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    # 在句子结尾填充，填充值为 </s> 对应的 id
    train_en = pad_sequences(train_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print('=' * 100)
    print('train_ch.shape:', train_ch.shape)
    print('train_en.shape:', train_en.shape)
    for i, x in enumerate(train_ch[0:5]):
        print('train_ch[{}]:'.format(i), x)
    for i, x in enumerate(train_en[0:5]):
        print('train_en[{}]:'.format(i), x)
    print()
elif mode == 'eval':
    # 在句子结尾填充，填充值为 </s> 对应的 id
    dev_ch = pad_sequences(dev_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    # 在句子结尾填充，填充值为 </s> 对应的 id
    dev_en = pad_sequences(dev_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print('=' * 100)
    print('dev_ch.shape:', dev_ch.shape)
    print('dev_en.shape:', dev_en.shape)
    for i, x in enumerate(dev_ch[0:5]):
        print('dev_ch[{}]:'.format(i), x)
    for i, x in enumerate(dev_en[0:5]):
        print('dev_en[{}]:'.format(i), x)
    print()
elif mode == 'infer':
    # 在句子结尾填充，填充值为 </s> 对应的 id
    test_ch = pad_sequences(test_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    # 在句子结尾填充，填充值为 </s> 对应的 id
    test_en = pad_sequences(test_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print('=' * 100)
    print('test_ch.shape:', test_ch.shape)
    print('test_en.shape:', test_en.shape)
    for i, x in enumerate(test_ch[0:5]):
        print('test_ch[{}]:'.format(i), x)
    for i, x in enumerate(test_en[0:5]):
        print('test_en[{}]:'.format(i), x)
    print()


"""
定义参数
"""
embedding_size = 512  # 嵌入层维度（词向量维度）
hidden_size = 512  # 隐藏层单元数
if mode == 'train':
    batch_size = 128
else:
    batch_size = 16
epochs = 20


# 定义参数的两种初始化方式
k_initializer = tf.contrib.layers.xavier_initializer()
e_initializer = tf.random_uniform_initializer(-1.0, 1.0)


# 定义了 4 个 placeholder
X = tf.placeholder(tf.int32, [None, maxlen_ch])
X_len = tf.placeholder(tf.int32, [None])
Y = tf.placeholder(tf.int32, [None, maxlen_en])
Y_len = tf.placeholder(tf.int32, [None])


# Y_in 是 decoder 阶段的输入，去掉结尾的 </s>
Y_in = Y[:, :-1]
# Y_out 是 decoder 阶段的输出，去掉开头的 <s>
Y_out = Y[:, 1:]


# 定义嵌入层（中文词），embedding
with tf.variable_scope('embedding_X'):
    embeddings_X = tf.get_variable('weights_X', shape=[len(word2id_ch), embedding_size], initializer=e_initializer)
    # embedded_X 的 shape 为：(batch_size, seq_len, embedding_size)
    embedded_X = tf.nn.embedding_lookup(embeddings_X, X)


# 定义嵌入层（英文词），embedding
with tf.variable_scope('embedding_Y'):
    embeddings_Y = tf.get_variable('weights_Y', shape=[len(word2id_en), embedding_size], initializer=e_initializer)
    embedded_Y = tf.nn.embedding_lookup(embeddings_Y, Y_in)


def single_cell(mode=mode):
    """
    定义 LSTM 单元
    @param mode: 模式（训练模式或者其它模式），如果是其它模式则不设置反向随机失活（dropout）
    @return cell: LSTM 单元
    """
    if mode == 'train':
        keep_prob = 0.8
    else:
        keep_prob = 1.0
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    # 添加 dropout
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell


def multi_cells(num_layers):
    """
    定义多层 LSTM
    @param num_layers: LSTM 层数
    @retrun tf.nn.rnn_cell.MultiRNNCell(cells): 返回多层 LSTM 神经网络
    """
    cells = []
    for i in range(num_layers):
        cell = single_cell()
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


"""
编码器部分，双向循环、单层 LSTM
"""
with tf.variable_scope('encoder'):
    num_layers = 1
    fw_cell = multi_cells(num_layers)
    bw_cell = multi_cells(num_layers)
    # sequence_length: 一个整数类型的向量，包含每个句子的真实长度
    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_X, dtype=tf.float32, sequence_length=X_len)
    # 62 个时间步，隐藏层单元数为 512，每个时间步都产生两个 512 维的数据，每个 batch 若干条数据
    # ((?, 62, 512), (?, 62, 512))
    print('=' * 100, 'bi_outputs:', bi_outputs)
    print()
    # 将每个时间步产生的两个 512 维的输出连接起来
    # (?, 62, 1024)
    encoder_outputs = tf.concat(bi_outputs, -1)
    print('=' * 100, 'encoder_outputs:', encoder_outputs)
    print()
    # (((c, h),),) ((c, h),))，如果有两层，则是 (((c, h), (c, h),), ((c, h), (c, h),))
    # c: (?, 512), h: (?, 512)
    print('=' * 100, 'bi_state:', bi_state)
    print()
    # encodere_state 的形式为: [(c, h), (c, h), …, (c, h)]，也就是所有层的 (c, h)
    encoder_state = []
    for i in range(num_layers):
        encoder_state.append(bi_state[0][i])  # forward
        encoder_state.append(bi_state[1][i])  # backward
    # 转为 tuple 形式
    encoder_state = tuple(encoder_state)
    print('=' * 100)
    for i in range(len(encoder_state)):
        print('encoder_state[{}]'.format(i), encoder_state[i])
    print()


"""
解码部分，两层 LSTM，注意力机制，
"""
with tf.variable_scope('decoder'):
    beam_width = 10
    memory = encoder_outputs
    if mode == 'infer':
        # 将 memory 中的每条数据复制 beam_width 份，在这里 beam_width 为 10，所以复制 10 份
        memory = tf.contrib.seq2seq.tile_batch(memory, beam_width)
        X_len_ = tf.contrib.seq2seq.tile_batch(X_len, beam_width)
        # 将 encoder_state 中的每条数据复制 10 份
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
        bs = batch_size * beam_width
    else:
        bs = batch_size
        X_len_ = X_len
    # 计算注意力权重的两种方案
    attention = tf.contrib.seq2seq.LuongAttention(hidden_size, memory, memory_sequence_length=X_len_, scale=True) # multiplicative
    # attention = tf.contrib.seq2seq.BahdanauAttention(hidden_size, memory, memory_sequence_length=X_len_, normalize=True) # additive
    # 两层 LSTM
    cell = multi_cells(num_layers * 2)
    # 添加注意力机制
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention, hidden_size, name='attention')
    # 解码部分的初始状态就是编码部分得到的 encoder_state
    decoder_initial_state = cell.zero_state(bs, tf.float32).clone(cell_state=encoder_state)
    # 定义输出层
    with tf.variable_scope('projected'):
        output_layer = tf.layers.Dense(len(word2id_en), use_bias=False, kernel_initializer=k_initializer)
    # infer 模式
    if mode == 'infer':
        start = tf.fill([batch_size], word2id_en['<s>'])
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, embeddings_Y, start, word2id_en['</s>'], decoder_initial_state, beam_width, output_layer)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, maximum_iterations=2 * tf.reduce_max(X_len))
        sample_id = outputs.predicted_ids
    else:
        # 训练阶段，使用 TrainingHelper + BasicDecoder 的组合，这一般是固定的，当然也可以自己定义 Helper 类，实现自己的功能
        helper = tf.contrib.seq2seq.TrainingHelper(embedded_Y, [maxlen_en - 1 for b in range(batch_size)])
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer)
        # 调用 dynamic_decode 进行解码，outputs 是一个 namedtuple，里面包含两项 (rnn_outputs, sample_id)
        # rnn_output: 由于设置 output_time_major 为 True，rnn_output 的 shape 为 [decoder_targets_length, batch_size, vocab_size]，保存 decode 每个时刻每个单词的概率，可以用来计算 loss
        # sample_id: [batch_size], tf.int32，保存最终的编码结果，可以表示最后的答案
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
        logits = outputs.rnn_output
        # shape=(128, ?, 20003)
        logits = tf.transpose(logits, (1, 0, 2))
        print('logits:', logits)


"""
不是 infer 模式时，计算 loss
"""
if mode != 'infer':
    with tf.variable_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_out, logits=logits)
        mask = tf.sequence_mask(Y_len, tf.shape(Y_out)[1], tf.float32)
        loss = tf.reduce_sum(loss * mask) / batch_size


"""
训练模式调整参数
"""
if mode == 'train':
    learning_rate = tf.Variable(0.0, trainable=False)
    params = tf.trainable_variables()
    # 梯度裁剪
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.0)
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads, params))


# 创建 session
sess = tf.Session()
# 初始化变量
sess.run(tf.global_variables_initializer())
if mode == 'train':
    # 创建 Saver 对象，用来保存模型
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_dir'
    # 如果文件夹不存在，创建文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    # 用于汇总标量数据
    tf.summary.scalar('loss', loss)
    # 用 tf.summary.merge_all 来整合所有的汇总操作
    summary = tf.summary.merge_all()
    # FileWriter：文件记录器，传入一个路径，FileWriter 将把数据写到这里。
    writer = tf.summary.FileWriter(OUTPUT_DIR)

    for e in range(epochs):
        total_loss = 0
        total_count = 0
        # 确定从第几个 epoch 开始降低学习率
        start_decay = int(epochs * 2 / 3)
        if e <= start_decay:
            lr = 1.0
        else:
            # 学习率衰减
            decay = 0.5 ** (int(4 * (e - start_decay) / (epochs - start_decay)))
            lr = 1.0 * decay
        sess.run(tf.assign(learning_rate, lr))

        # 打乱数据
        train_ch, len_train_ch, train_en, len_train_en = shuffle(train_ch, len_train_ch, train_en, len_train_en)

        # tqdm 可以封装一个进度条，查看进度信息
        for i in tqdm(range(train_ch.shape[0] // batch_size)):
            X_batch = train_ch[i * batch_size: i * batch_size + batch_size]
            X_len_batch = len_train_ch[i * batch_size: i * batch_size + batch_size]
            Y_batch = train_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = len_train_en[i * batch_size: i * batch_size + batch_size]
            # 每个长度减 1
            Y_len_batch = [l - 1 for l in Y_len_batch]
            # 传入数据
            feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
            _, ls_ = sess.run([optimizer, loss], feed_dict=feed_dict)
            # 计算每个 batch 的总 loss 值
            total_loss += ls_ * batch_size
            # 计算每个 batch 的输出的总长度
            total_count += np.sum(Y_len_batch)
            # 每 100 步记录一下训练过程
            if i > 0 and i % 100 == 0:
                writer.add_summary(sess.run(summary, feed_dict=feed_dict), e * train_ch.shape[0] // batch_size + i)
                writer.flush()
        print('Epoch %d lr %.3f perplexity %.2f' % (e, lr, np.exp(total_loss / total_count)))
        saver.save(sess, os.path.join(OUTPUT_DIR, 'nmt'))


if mode == 'eval':
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_dir'
    saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))
    total_loss = 0
    total_count = 0
    for i in tqdm(range(dev_ch.shape[0] // batch_size)):
        X_batch = dev_ch[i * batch_size: i * batch_size + batch_size]
        X_len_batch = len_dev_ch[i * batch_size: i * batch_size + batch_size]
        Y_batch = dev_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = len_dev_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = [l - 1 for l in Y_len_batch]
        feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
        ls_ = sess.run(loss, feed_dict=feed_dict)
        total_loss += ls_ * batch_size
        total_count += np.sum(Y_len_batch)
    print('Dev perplexity %.2f' % np.exp(total_loss / total_count))


if mode == 'infer':
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_dir'
    saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))

    def translate(ids):
        words = [id2word_en[i] for i in ids]
        if words[0] == '<s>':
            words = words[1:]
        if '</s>' in words:
            words = words[:words.index('</s>')]
        return ' '.join(words)

    fw = open('output_test', 'w')
    for i in tqdm(range(test_ch.shape[0] // batch_size)):
        X_batch = test_ch[i * batch_size: i * batch_size + batch_size]
        X_len_batch = len_test_ch[i * batch_size: i * batch_size + batch_size]
        Y_batch = test_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = len_test_en[i * batch_size: i * batch_size + batch_size]
        Y_len_batch = [l - 1 for l in Y_len_batch]
        feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
        ids = sess.run(sample_id, feed_dict=feed_dict)
        ids = np.transpose(ids, (1, 2, 0))
        ids = ids[:, 0, :]
        for j in range(ids.shape[0]):
            sentence = translate(ids[j])
            fw.write(sentence + '\n')
    fw.close()

    # from nmt.utils.evaluation_utils import evaluate
    # for metric in ['bleu', 'rouge']:
    #     score = evaluate('data/test.en', 'output_test', metric)
    #     print(metric, score / 100)
