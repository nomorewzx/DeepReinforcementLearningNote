# coding=utf-8
"""
Add summary for pong model, and change loss func to log prob.
"""
import numpy as np
import tensorflow as tf
import cPickle as pickle
import pong_utils
import gym
import os

HIDDEN_LAYER_COUNTS = 200
LEARNING_RATE = 1e-3

REWARD_DISCOUNT_FACTOR = 0.99

RESUME = False
RENDER = True

INPUT_DIM = 80 * 80
ACTION_SPACE_SIZE = 3

STILL = 1
MOVE_UP = 2
MOVE_DOWN = 3
DECAY = 0.99

ACTION_SPACE = [STILL, MOVE_UP, MOVE_DOWN]

CHECK_POINT_DIR = "saved_model_cross_entropy_loss/"

SUMMARY_DIR = "summary/cross_entropy_loss/"

graph = tf.Graph()

def inference_model(input_statuses):
    # 神经网络模型，input_statues 是长度为n的输入，经过神经网络模型推导，得到 action 概率向量
    # logits = (relu((x * w1)) * w2)
    # action_probs = softmax(logits)
    # 其中softmax 操作即是将向量中的任意元素转换为 0 - 0.1 且各元素和为1.0
    hidden_layer_output = tf.nn.relu(tf.matmul(input_statuses, hidden_layer_weights))
    sampled_policy_logit = tf.matmul(hidden_layer_output, output_layer_weights)
    action_probs = tf.nn.softmax(sampled_policy_logit)

    return action_probs


def build_loss_func(logits, rewards):
    # 构建 loss(损失) 函数，通过求loss函数的最小值，即可达到更新神经网络权重矩阵参数的目的
    #  loss = - labels * log(logits) * rewards
    # 此时loss 为向量/矩阵，经过求和操作，可得到一个实数, 如下:
    # loss = sum(loss)

    advantaged_cross_entropy = (-tf.multiply(labels, tf.log(logits))) * rewards
    loss = tf.reduce_sum(advantaged_cross_entropy)

    return loss


def build_optimizer(loss, global_step):

    # 使用RMSProp优化器求 loss函数的最小值，其中global_step 为整数，minimize(loss) 函数每运行一次，global_step 即 +1
    train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY).minimize(loss, global_step= global_step)
    return train_op


with graph.as_default():

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # 定义图模型的输入, input_statuses : 经过处理后的游戏局势图片
    # labels: 在游戏进行中，agent 采取的所有action的列表
    # rewards: agent每次action后，env所给的 reward(奖赏)值
    # discounted_rewards: 经过处理后的reward(奖赏)值
    # 注: agent的行为是: 观察每个游戏局势图片，做出action，因此得到reward, 因此这些输入需要一一对应

    input_statuses = tf.placeholder(shape=[None, INPUT_DIM], dtype=tf.float32, name="input_status_by_episode")
    labels = tf.placeholder(shape=[None, ACTION_SPACE_SIZE], dtype=tf.float32, name="labels")
    discounted_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="discounted_rewards_by_episode")
    rewards = tf.placeholder(shape=[None, 1], dtype= tf.float32, name="rewards_by_episode")

    # 使用 xavier 分布初始化神经网络权重矩阵, 两个矩阵的尺寸分别为 [6400, 200] 以及 [200, 3]
    hidden_layer_weights = tf.Variable(tf.truncated_normal([INPUT_DIM, HIDDEN_LAYER_COUNTS],mean=0.0, stddev= 1. / np.sqrt(INPUT_DIM), dtype=tf.float32), name="hidden_layer_weights")
    output_layer_weights = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_COUNTS, ACTION_SPACE_SIZE], mean=0.0, stddev= 1. / np.sqrt(HIDDEN_LAYER_COUNTS), dtype=tf.float32), name="output_layer_weights")

    # 输入当前局势 input_statues, 得出 action 概率向量, 例: [0.1, 0.2, 0.7]
    action_probs = inference_model(input_statuses)

    # 输入当前局势 input_statues, 得出 action 概率向量, 例: [0.1, 0.2, 0.7]
    logits = inference_model(input_statuses)

    # 根据 以上得出的概率向量logits，以及labels, 构建 loss 函数
    loss = build_loss_func(logits, discounted_rewards)

    # 使用优化器求loss函数的最小值
    train_op = build_optimizer(loss, global_step)

    # 使用summary操作，存储训练过程中的指标: loss数值，以及每局游戏agent所得分数, 以便观测训练过程
    loss_summary = tf.summary.scalar("loss", loss)
    reward_summary = tf.summary.scalar("reward", tf.reduce_sum(rewards))

    summary_op = tf.summary.merge_all()

# 运行之前定义好的图模型 (graph)，需要打开一个session
with tf.Session(graph=graph) as session:

    # 将观测值写入文件, 文件地址由 SUMMARY_DIR指定
    writer = tf.summary.FileWriter(SUMMARY_DIR, graph= session.graph)

    # 创建saver，以保存图模型以及训练过程中的参数，max_to_keep 指定最多保存前4步的训练结果
    saver = tf.train.Saver(max_to_keep= 4)

    # 初始化图模型中的所有变量, 该命令必须运行
    tf.global_variables_initializer().run()

    # 指定存储图模型以及训练过程参数的地址, 如果该地址中已有存储的模型，就从该模型中恢复训练过程ß
    check_point = tf.train.get_checkpoint_state(os.path.dirname(CHECK_POINT_DIR + 'checkpoint'))
    if check_point and check_point.model_checkpoint_path:
        saver.restore(session, check_point.model_checkpoint_path)

    # 创建并重置 Pong-v0 环境
    pong_env = gym.make("Pong-v0")
    observation = pong_env.reset()

    # 指定 前一个 status
    prev_status = None

    # 控制台输出值，以便从控制台观测训练过程
    reward_sum = 0
    game_numer = 0

    # 初始化三个数组，以存储 游戏盘面状态，根据状态所采取的action，以及action所得到的reward
    input_status_list = list()
    one_hot_action_list = list()
    reward_list = list()

    while True:
        if RENDER:
            pong_env.render()

        # 将游戏整盘面处理为只包含重点区域，且为灰度图的新图像
        current_status = pong_utils.preprocessing_observation_of_pong(observation)

        # 将当前 status 与 之前的status取差值，以使模型能观测到两次status之间局势的变化
        if prev_status is not None:
            status = current_status - prev_status
        else:
            status = np.zeros(INPUT_DIM)

        prev_status = current_status

        # 将status 推入 status列表
        input_status_list.append(status)

        # 将当前status输入模型,得到action概率向量
        action_probs_train = session.run(action_probs, feed_dict={input_statuses: [status]})
        action_probs_train = action_probs_train[0, :]

        # 使用np.random.choice函数选出一个action
        action_index = np.random.choice(ACTION_SPACE_SIZE, p= action_probs_train)

        one_hot_action = np.zeros_like(action_probs_train)
        one_hot_action[action_index] = 1

        # 将选出的action推入action 数组
        one_hot_action_list.append(one_hot_action)

        # agent 执行选出的action，得到新的游戏盘面, reward, 以及标示当前游戏局是否结束的flag: done
        observation, reward, done, info = pong_env.step(ACTION_SPACE[action_index])

        # 将得到的reward推入 reward 列表
        reward_list.append(reward)

        # 更新reward_sum
        reward_sum += reward

        if done:
            # 一局游戏结束，将rewards 进行处理, 得到discounted_reward_list
            discounted_reward_list = pong_utils.discounted_rewards(reward_list=reward_list, gamma=REWARD_DISCOUNT_FACTOR)

            # 用于记录一局游戏中，有几次胜负
            game_numer = 0

            # 转换 以上各list的形状，以使这些数据符合图模型的输入要求
            episode_rewards = np.vstack(reward_list)
            episode_discounted_rewards = np.vstack(discounted_reward_list)
            episode_labels = np.array(one_hot_action_list, dtype=np.float32)
            episode_statuses = np.array(input_status_list, dtype=np.float32)

            # 构建feed_dict，feed_dict 的key是图模型的placeholder, value是要传给这些placeholder的value
            feed_dict = {input_statuses: input_status_list, discounted_rewards: episode_discounted_rewards, rewards: episode_rewards, labels: episode_labels}

            # 使用sessin.run() 运行图模型中的节点, 同时用feed_dict传入参数, 执行完 train_op 就是使用一盘游戏的数据训练了一次模型
            _, loss_train, logits_train, global_step_train, summary_op_train = session.run([train_op, loss, logits, global_step, summary_op], feed_dict=feed_dict)
            print("Resetting env. episode reward total was %f . Loss is %f" % (reward_sum, loss_train))
            reward_sum = 0
            # 保存图模型以及训练过程中的参数
            saver.save(session, CHECK_POINT_DIR + 'pong_model', global_step= global_step_train)

            # 因一盘游戏已结束，要开始下一盘游戏，需要重置游戏环境
            observation = pong_env.reset()  # reset env
            prev_status = None

            # 存储改次训练结束后的观测指标
            writer.add_summary(summary_op_train, global_step= global_step_train)

            # 重置各数组
            input_status_list = []
            reward_list = []
            one_hot_action_list = []

        # 输入每次有得分时的情况
        if reward != 0:
            game_numer +=1
            print("Episode %d game %d finished, reward: %f" % (global_step.eval(), game_numer, reward) + (" " if reward == -1 else "!!!!!!!!!!"))