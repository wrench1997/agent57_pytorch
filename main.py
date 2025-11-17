import argparse
import os
import shutil
import time
import pickle
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from tester import Tester
from buffer import SegmentReplayBuffer
from learner import Learner
from utils import seed_evrything


weight_dir = "result/weights"
os.makedirs(weight_dir, exist_ok=True)

def main(args):
    if os.path.exists("log"):
        shutil.rmtree("log")
    os.makedirs("log")

    seed_evrything(args.seed)

    total_s = time.time()
    in_q_loss_history, ex_q_loss_history, embed_loss_history, lifelong_loss_history, score_history = [], [], [], [], []
    
    # 初始化Learner
    learner = Learner(env_name=args.env_name,
                      target_update_period=args.target_update_period,
                      eta=args.eta,
                      n_frames=args.n_frames,
                      num_arms=args.num_arms,
                      lamda=args.lamda,
                      burnin_length=args.burnin_length,
                      unroll_length=args.unroll_length,
                      in_q_lr=args.in_q_lr,
                      ex_q_lr=args.ex_q_lr,
                      embed_lr=args.embed_lr,
                      lifelong_lr=args.lifelong_lr,
                      in_q_clip_grad=args.in_q_clip_grad,
                      ex_q_clip_grad=args.ex_q_clip_grad,
                      embed_clip_grad=args.embed_clip_grad,
                      lifelong_clip_grad=args.lifelong_clip_grad)
    
    in_q_weight, ex_q_weight, embed_weight, trained_lifelong_weight, original_lifelong_weight = learner.define_network()
    
    # 初始化Agents
    agents = [Agent(pid=i,
                    env_name=args.env_name,
                    n_frames=args.n_frames,
                    epsilon=args.epsilon_l ** (1 + args.alpha_l * i / (args.num_agents - 1)),
                    eta=args.eta,
                    lamda=args.lamda,
                    agent_update_period=args.agent_update_period,
                    num_rollout=args.num_rollout,
                    num_arms=args.num_arms,
                    k=args.k,
                    L=args.L,
                    burnin_length=args.burnin_length,
                    unroll_length=args.unroll_length,
                    window_size=args.window_size,
                    ucb_epsilon=args.ucb_epsilon,
                    ucb_beta=args.ucb_beta,
                    original_lifelong_weight=original_lifelong_weight)
              for i in range(args.num_agents)]

    replay_buffer = SegmentReplayBuffer(buffer_size=args.buffer_size, weight_expo=args.weight_expo)   

    # 初始化Tester
    tester = Tester(env_name=args.env_name,
                    n_frames=args.n_frames,
                    num_arms=args.num_arms,
                    L=args.L,
                    k=args.k,
                    window_size=args.window_size,
                    ucb_epsilon=args.ucb_epsilon,
                    ucb_beta=args.ucb_beta,
                    switch_test_cycle=args.switch_test_cycle,
                    original_lifelong_weight=original_lifelong_weight)

    # 使用线程池来并行执行Agent的rollout
    with ThreadPoolExecutor(max_workers=args.num_agents) as executor:
        # 初始填充经验池
        for i in range(args.n_agent_burnin):
            s = time.time()
            
            # 随机选择一个agent
            agent_idx = i % args.num_agents
            priorities, segments, pid = agents[agent_idx].sync_weights_and_rollout(
                in_q_weight=in_q_weight,
                ex_q_weight=ex_q_weight,
                embed_weight=embed_weight,
                lifelong_weight=trained_lifelong_weight
            )
            
            replay_buffer.add(priorities, segments)
            
            with open(f"log/agent_time_check.txt", mode="a") as f:
                f.write(f"{i}th Agent's time[sec]: {time.time() - s:.5f}\n")

    print("="*100)
    
    learner_cycles = 1
    agent_cycles = 0
    n_segment_added = 0

    while learner_cycles <= args.n_learner_cycle:
        agent_cycles += 1
        s = time.time()
        
        # 并行收集经验
        futures = []
        with ThreadPoolExecutor(max_workers=args.num_agents) as executor:
            for agent_idx in range(args.num_agents):
                future = executor.submit(
                    agents[agent_idx].sync_weights_and_rollout,
                    in_q_weight=in_q_weight,
                    ex_q_weight=ex_q_weight,
                    embed_weight=embed_weight,
                    lifelong_weight=trained_lifelong_weight
                )
                futures.append(future)
            
            for future in futures:
                priorities, segments, pid = future.result()
                replay_buffer.add(priorities, segments)
                n_segment_added += len(segments)
        
        # 从经验池中采样并更新网络
        minibatchs = [replay_buffer.sample_minibatch(batch_size=args.batch_size) for _ in range(args.update_iter)]
        in_q_weight, ex_q_weight, embed_weight, trained_lifelong_weight, indices, priorities, in_q_loss, ex_q_loss, embed_loss, lifelong_loss = learner.update_network(minibatchs)
        
        replay_buffer.update_priority(indices, priorities)
        
        with open(f"log/loss_history.txt", mode="a") as f:
            f.write(f"{learner_cycles}th results => Agent cycle: {agent_cycles}, Added: {n_segment_added}, InQLoss: {in_q_loss:.4f}, ExQLoss: {ex_q_loss:.4f}, EmbeddingLoss: {embed_loss:.4f}, LifeLongLoss: {lifelong_loss:.8f} \n")

        in_q_loss_history.append((learner_cycles-1, in_q_loss))
        ex_q_loss_history.append((learner_cycles-1, ex_q_loss))
        embed_loss_history.append((learner_cycles-1, embed_loss))
        lifelong_loss_history.append((learner_cycles-1, lifelong_loss))

        # 测试当前策略
        test_score = tester.test_play(in_q_weight=in_q_weight,
                                     ex_q_weight=ex_q_weight,
                                     embed_weight=embed_weight,
                                     lifelong_weight=trained_lifelong_weight)
        
        if test_score is not None:
            score_history.append((learner_cycles-args.switch_test_cycle, test_score))
            with open(f"log/score_history.txt", mode="a") as f:
                f.write(f"Cycle: {learner_cycles}, Score: {test_score}\n")

        if learner_cycles % args.freq_weight_save == 0:
            learner.save(weight_dir, learner_cycles)

        learner_cycles += 1
        agent_cycles = 0
        n_segment_added = 0
        s = time.time()

    wallclocktime = round(time.time() - total_s, 2)
    cycles, scores = zip(*score_history)
    plt.plot(cycles, scores)
    plt.title(f"total time: {wallclocktime} sec")
    plt.ylabel(f"test_score")
    plt.savefig(f"log/history_{args.num_agents}agents.png")
    plt.close()

    cycles, loss = zip(*in_q_loss_history)
    plt.plot(cycles, loss)
    plt.title(f"total time: {wallclocktime} sec \n Intrinsic Loss")
    plt.ylabel(f"intrinsic loss")
    plt.savefig(f"log/intrinsic_loss_history_{args.num_agents}agents.png")
    plt.close()

    cycles, loss = zip(*ex_q_loss_history)
    plt.plot(cycles, loss)
    plt.title(f"total time: {wallclocktime} sec \n Extrinsic Loss")
    plt.ylabel(f"extrinsic loss")
    plt.savefig(f"log/extrinsic_loss_history_{args.num_agents}agents.png")
    plt.close()

    cycles, loss = zip(*embed_loss_history)
    plt.plot(cycles, loss)
    plt.title(f"total time: {wallclocktime} sec \n Embedding Loss")
    plt.ylabel(f"embedding loss")
    plt.savefig(f"log/embedding_loss_history_{args.num_agents}agents.png")
    plt.close()

    cycles, loss = zip(*lifelong_loss_history)
    plt.plot(cycles, loss)
    plt.title(f"total time: {wallclocktime} sec \n LifeLong Loss")
    plt.ylabel(f"lifelong loss")
    plt.savefig(f"log/lifelong_loss_history_{args.num_agents}agents.png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agent57')
    # Agent
    parser.add_argument('--num_agents', default=2, type=int)
    parser.add_argument('--agent_update_period', default=100, type=int)
    parser.add_argument('--num_rollout', default=10, type=int)
    parser.add_argument('--epsilon_l', default=0.4, type=float)
    parser.add_argument('--alpha_l', default=8, type=int)
    # Tester
    parser.add_argument('--switch_test_cycle', default=10, type=int)
    # Learner
    parser.add_argument('--target_update_period', default=10, type=int)
    parser.add_argument('--in_q_lr', default=1e-4, type=float)
    parser.add_argument('--ex_q_lr', default=1e-4, type=float)
    parser.add_argument('--embed_lr', default=5e-4, type=float)
    parser.add_argument('--lifelong_lr', default=5e-4, type=float)
    parser.add_argument('--in_q_clip_grad', default=40, type=float)
    parser.add_argument('--ex_q_clip_grad', default=40, type=float)
    parser.add_argument('--embed_clip_grad', default=40, type=float)
    parser.add_argument('--lifelong_clip_grad', default=40, type=float)
    # buffer
    parser.add_argument('--buffer_size', default=2**16, type=int)
    parser.add_argument('--weight_expo', default=0.0, type=float)
    # UCB
    parser.add_argument('--window_size', default=90, type=int)
    parser.add_argument('--ucb_epsilon', default=0.5, type=float)
    parser.add_argument('--ucb_beta', default=1.0, type=float)
    # Intrinsic
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--L', default=5, type=int)
    parser.add_argument('--num_arms', default=32, type=int)
    # Base
    parser.add_argument('--seed', default=0, type=int)    
    parser.add_argument('--gamma', default=0.997, type=float)
    parser.add_argument('--eta', default=0.9, type=float)
    parser.add_argument('--lamda', default=0.95, type=float)
    parser.add_argument('--n_frames', default=4, type=int)
    parser.add_argument('--env_name', default="ALE/Breakout-v5")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--update_iter', default=4, type=int)
    parser.add_argument('--burnin_length', default=20, type=int)
    parser.add_argument('--unroll_length', default=20, type=int)
    parser.add_argument('--n_agent_burnin', default=16, type=int)
    parser.add_argument('--n_learner_cycle', default=5000, type=int)
    parser.add_argument('--freq_weight_save', default=100, type=int)
    main(parser.parse_args())