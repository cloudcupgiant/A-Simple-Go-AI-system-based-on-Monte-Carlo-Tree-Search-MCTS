import math
import random
import time

from dlgo.goboard import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result

__all__ = ["MCTSAgent"]


class MCTSNode:
    """MCTS 搜索树中的节点。"""

    def __init__(self, game_state, parent=None, parent_move=None, prior=1.0):
        # 当前节点对应的棋局状态
        self.game_state = game_state
        # 父节点
        self.parent = parent
        # 从父节点走到当前节点的动作
        self.parent_move = parent_move
        # 子节点列表
        self.children = []
        # 节点访问次数
        self.visit_count = 0
        # 节点累计价值
        self.value_sum = 0.0
        # 先验（预留字段）
        self.prior = prior
        # 未展开动作：过滤 resign，减少噪声
        self._unexpanded_moves = [
            m for m in game_state.legal_moves() if not m.is_resign
        ]

    @property
    def value(self):
        """节点平均价值。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶子节点。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def is_fully_expanded(self):
        """是否已展开所有候选动作。"""
        return len(self._unexpanded_moves) == 0

    def best_child(self, c=1.414):
        """按 UCT 公式选择最优子节点。"""
        if not self.children:
            return None

        parent_visits = max(1, self.visit_count)
        log_parent = math.log(parent_visits)

        best_score = -float("inf")
        best_nodes = []

        for child in self.children:
            # 未访问节点优先探索
            if child.visit_count == 0:
                score = float("inf")
            else:
                # UCT = Q + c * sqrt(ln(N) / n)
                exploit = child.value
                explore = c * math.sqrt(log_parent / child.visit_count)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        # 并列最优时随机打破平局
        return random.choice(best_nodes)

    def expand(self, score_fn=None, top_k=3):
        """展开一个动作并返回新子节点。"""
        if not self._unexpanded_moves:
            return None

        if score_fn is None:
            idx = random.randrange(len(self._unexpanded_moves))
            move = self._unexpanded_moves.pop(idx)
        else:
            # 启发式打分后从 top-k 中按权重采样
            scored = [
                (score_fn(self.game_state, m), m)
                for m in self._unexpanded_moves
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            k = max(1, min(top_k, len(scored)))
            candidates = scored[:k]
            min_score = min(score for score, _ in candidates)
            weights = [(score - min_score) + 1e-3 for score, _ in candidates]
            moves = [move for _, move in candidates]

            move = random.choices(moves, weights=weights, k=1)[0]
            self._unexpanded_moves.remove(move)

        next_state = self.game_state.apply_move(move)
        child = MCTSNode(next_state, parent=self, parent_move=move)
        self.children.append(child)
        return child

    def backup(self, value):
        """将模拟结果从当前节点回传到根节点。"""
        node = self
        # 视角翻转：当前节点保存的是“到达该节点落子方”的价值
        node_value = 1.0 - value

        while node is not None:
            node.visit_count += 1
            node.value_sum += node_value
            # 双人零和，逐层翻转视角
            node_value = 1.0 - node_value
            node = node.parent


class MCTSAgent:
    """5x5 围棋 MCTS 智能体。"""

    def __init__(
        self,
        num_rounds=640,
        temperature=1.414,
        max_seconds=1,
        rollout_max_depth=17,
        rollout_top_k=4,
    ):
        # 每步最大模拟轮数
        self.num_rounds = num_rounds
        # UCT 探索系数
        self.temperature = temperature
        # 每步思考时间预算（秒）
        self.max_seconds = max_seconds
        # rollout 最大深度（优化 1）
        self.rollout_max_depth = rollout_max_depth
        # rollout 采样时使用的 top-k（优化 2）
        self.rollout_top_k = rollout_top_k

    def select_move(self, game_state: GameState) -> Move:
        """MCTS 主循环：选择 -> 扩展 -> 模拟 -> 回传。"""
        start_ts = time.time()
        rounds = 0
        chosen_move = Move.pass_turn()

        try:
            if game_state.is_over():
                return chosen_move

            if not any(m.is_play for m in game_state.legal_moves()):
                return chosen_move

            root = MCTSNode(game_state)
            if not root._unexpanded_moves:
                return chosen_move

            deadline = time.time() + self.max_seconds

            while rounds < self.num_rounds and time.time() < deadline:
                node = root

                # 1) Selection
                while (
                    not node.is_terminal()
                    and node.is_fully_expanded()
                    and node.children
                ):
                    node = node.best_child(self.temperature)

                # 2) Expansion
                if not node.is_terminal() and not node.is_fully_expanded():
                    child = node.expand(self._expand_move_score, top_k=3)
                    if child is not None:
                        node = child

                # 3) Simulation
                value = self._simulate(node.game_state)

                # 4) Backpropagation
                node.backup(value)
                rounds += 1

            chosen_move = self._select_best_move(root)
            return chosen_move
        finally:
            # 统一计时日志：用于评估 max_seconds 是否足够
            think_time = time.time() - start_ts
            print(
                "[MCTS-TIME] player=%s think_time=%.3fs rounds=%d max_seconds=%.3fs"
                % (
                    game_state.next_player.name,
                    think_time,
                    rounds,
                    self.max_seconds,
                )
            )

    def _simulate(self, game_state):
        """
        Rollout 策略（两种提速）：
        1) 深度限制
        2) 启发式走子
        """
        rollout_state = game_state
        start_player: Player = rollout_state.next_player
        consecutive_passes = 0

        for _ in range(self.rollout_max_depth):
            if rollout_state.is_over():
                break

            legal_moves = [
                m for m in rollout_state.legal_moves() if not m.is_resign
            ]
            if not legal_moves:
                break

            move = self._select_rollout_move(rollout_state, legal_moves)
            if move.is_pass:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            rollout_state = rollout_state.apply_move(move)
            if consecutive_passes >= 2:
                break

        # 未终局时按当前盘面计分估值
        if rollout_state.is_over():
            winner = rollout_state.winner()
        else:
            winner = compute_game_result(rollout_state).winner

        if winner is None:
            return 0.5
        return 1.0 if winner == start_player else 0.0

    def _select_rollout_move(self, game_state, legal_moves):
        """按启发式评分在候选动作中采样 rollout 动作。"""
        play_moves = [m for m in legal_moves if m.is_play]
        if not play_moves:
            return Move.pass_turn()

        scored_moves = [
            (self._rollout_move_score(game_state, move), move)
            for move in play_moves
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        top_k = max(1, min(self.rollout_top_k, len(scored_moves)))
        candidates = scored_moves[:top_k]

        min_score = min(score for score, _ in candidates)
        weights = [(score - min_score) + 1e-3 for score, _ in candidates]
        candidate_moves = [move for _, move in candidates]

        # 收官期提高 pass 概率，减少无效拖局
        if len(play_moves) <= 4 and random.random() < 0.30:
            return Move.pass_turn()
        if random.random() < 0.02:
            return Move.pass_turn()

        return random.choices(candidate_moves, weights=weights, k=1)[0]

    def _rollout_move_score(self, game_state, move):
        """rollout 启发式评分：中心偏好 + 邻接战术特征。"""
        board = game_state.board
        point: Point = move.point
        current_player: Player = game_state.next_player
        score = 0.0

        # 轻微中心偏好
        center_row = (board.num_rows + 1) / 2.0
        center_col = (board.num_cols + 1) / 2.0
        dist_to_center = abs(point.row - center_row) + abs(point.col - center_col)
        score += 2.0 - 0.35 * dist_to_center

        # 局部战术特征
        friendly_neighbors = 0
        enemy_neighbors = 0
        atari_targets = 0

        for neighbor in point.neighbors():
            if not board.is_on_grid(neighbor):
                continue
            stone = board.get(neighbor)
            if stone is None:
                continue
            if stone == current_player:
                friendly_neighbors += 1
            else:
                enemy_neighbors += 1
                enemy_string = board.get_go_string(neighbor)
                if enemy_string is not None and enemy_string.num_liberties == 1:
                    atari_targets += 1

        score += 0.35 * friendly_neighbors
        score += 0.55 * enemy_neighbors
        score += 1.5 * atari_targets

        # 轻微随机扰动，避免策略过早固化
        score += random.random() * 0.05
        return score

    def _select_best_move(self, root):
        """从根节点子节点中选最终动作。"""
        if not root.children:
            return Move.pass_turn()

        # 主要按访问次数选，value 作为次关键字
        best_child = max(root.children, key=lambda c: (c.visit_count, c.value))

        # 若 pass 分支访问量接近最优，也允许主动收官
        pass_child = next((c for c in root.children if c.parent_move.is_pass), None)
        if pass_child is not None and pass_child.visit_count >= 0.9 * best_child.visit_count:
            return pass_child.parent_move

        return best_child.parent_move

    def _expand_move_score(self, game_state, move):
        """扩展阶段的启发式评分函数。"""
        if move.is_pass:
            return -1000.0

        board = game_state.board
        point = move.point
        current_player = game_state.next_player
        score = 0.0

        # 中心偏好（比 rollout 略弱）
        center_row = (board.num_rows + 1) / 2.0
        center_col = (board.num_cols + 1) / 2.0
        dist_to_center = abs(point.row - center_row) + abs(point.col - center_col)
        score += 1.2 - 0.25 * dist_to_center

        # 邻接特征
        friendly_neighbors = 0
        enemy_neighbors = 0
        capture_targets = 0
        save_targets = 0

        for neighbor in point.neighbors():
            if not board.is_on_grid(neighbor):
                continue
            stone = board.get(neighbor)
            if stone is None:
                continue

            string = board.get_go_string(neighbor)
            if stone == current_player:
                friendly_neighbors += 1
                if string is not None and string.num_liberties == 1:
                    save_targets += 1
            else:
                enemy_neighbors += 1
                if string is not None and string.num_liberties == 1:
                    capture_targets += 1

        score += 0.30 * friendly_neighbors
        score += 0.40 * enemy_neighbors
        score += 3.0 * capture_targets
        score += 2.2 * save_targets

        # 奖励落子后自身气数
        next_state = game_state.apply_move(move)
        new_string = next_state.board.get_go_string(point)
        if new_string is not None:
            score += 0.45 * new_string.num_liberties

        return score
