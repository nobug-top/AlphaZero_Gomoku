# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
from rich.console import Console
import os
import random
import time
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer

# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras

console = Console()


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        # 棋盘
        self.board_width = 10
        self.board_height = 10
        # 连成五子
        self.n_in_row = 5
        self.board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row,
        )
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        # 训练数据缓冲区大小
        self.buffer_size = 10000
        # 每批次训练数据大小
        self.batch_size = 1024
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # 停止相关
        # early_stop_patience 次训练的 提升小于 early_stop_delta 则停止训练
        self.early_stop_patience = 5
        self.early_stop_delta = 0.01
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                model_file=init_model,
            )
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
            )
        console.print(
            f"[bold green] Training device: {self.policy_value_net.device}",
        )
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i
                )
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(
                self.mcts_player, temp=self.temp
            )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier,
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()
        ) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()
        ) / np.var(np.array(winner_batch))
        console.print(
            (
                "[cyan]kl:{:.5f}[/], "
                "lr_mult:{:.3f}, "
                "[yellow]loss:{:.4f}[/], "
                "entropy:{:.4f}, "
                "exp_var_old:{:.3f}, "
                "exp_var_new:{:.3f}"
            ).format(
                kl,
                self.lr_multiplier,
                float(loss),
                float(entropy),
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
        )
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0
            )
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        console.print(
            "[bold]eval:[/] num_playouts:{}, win:{}, lose:{}, tie:{}, win_ratio:{:.3f}".format(
                self.pure_mcts_playout_num,
                win_cnt[1],
                win_cnt[2],
                win_cnt[-1],
                win_ratio,
            )
        )
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            no_improve_count = 0
            best_for_stop = 0.0
            for i in range(self.game_batch_num):
                batch_start = time.perf_counter()
                self.collect_selfplay_data(self.play_batch_size)
                selfplay_elapsed = time.perf_counter() - batch_start
                train_elapsed = 0.0
                loss = None
                entropy = None
                if len(self.data_buffer) > self.batch_size:
                    train_start = time.perf_counter()
                    loss, entropy = self.policy_update()
                    train_elapsed = time.perf_counter() - train_start
                console.print(
                    (
                        f"batch i:{i + 1}, episode_len:{self.episode_len}, "
                        f"buffer:{len(self.data_buffer)}/{self.buffer_size}, "
                        f"[bold blue]selfplay:{selfplay_elapsed:.2f}s[/], "
                        f"[bold magenta]train:{train_elapsed:.2f}s[/]"
                    )
                )
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    console.print(f"[bold]current self-play batch:[/] {i + 1}")
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model("./current_policy.model")
                    if win_ratio > self.best_win_ratio:
                        console.print("[bold green]New best policy!!!!!!!![/]")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model("./best_policy.model")
                        if (
                            self.best_win_ratio == 1.0
                            and self.pure_mcts_playout_num < 5000
                        ):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                    if self.early_stop_patience > 0:
                        if win_ratio > best_for_stop + self.early_stop_delta:
                            best_for_stop = win_ratio
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                            console.print(
                                "early-stop check: no_improve_count={}/{}".format(
                                    no_improve_count,
                                    self.early_stop_patience,
                                )
                            )
                            if no_improve_count >= self.early_stop_patience:
                                console.print(
                                    "Early stop: win_ratio no longer improving"
                                )
                                break
        except KeyboardInterrupt:
            print("\n\rquit")


if __name__ == "__main__":
    init_model = None
    if os.path.exists("./current_policy.model"):
        init_model = "./current_policy.model"
        console.print("[bold yellow]Resuming from current_policy.model[/]")
    elif os.path.exists("./best_policy.model"):
        init_model = "./best_policy.model"
        console.print("[bold yellow]No current_policy found, resuming from best_policy.model[/]")
    else:
        console.print("[bold yellow]No existing model found, start training from scratch[/]")

    training_pipeline = TrainPipeline(init_model=init_model)
    training_pipeline.run()
