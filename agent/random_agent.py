"""
第一小问（必做）：随机 AI

基于随机落子（但需满足规则）的基础围棋 AI
用于验证规则调用和基础设施正常工作。
"""

import random
from dlgo import GameState, Move

__all__ = ["RandomAgent"]


class RandomAgent:
    """
    随机落子智能体 - 第一小问实现

    从所有合法棋步中均匀随机选择，包括：
    - 正常落子
    - 停一手 (pass)
    - 认输 (resign)
    """

    def __init__(self):
        pass

    def select_move(self, game_state: GameState) -> Move:
        """
        选择随机合法棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            随机选择的合法 Move
        """
        play_moves = []  # 行为列表
        for move in game_state.legal_moves():  #遍历合法步
            if move.is_play: # 添加有point的步
                play_moves.append(move)

        if play_moves:
            return random.choice(play_moves)
        return Move.pass_turn() #无棋可下时


def random_agent(game_state: GameState) -> Move:
    """函数接口，兼容 play.py 的调用方式"""
    agent = RandomAgent()
    return agent.select_move(game_state)
