"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move
from dlgo.scoring import compute_game_result
import time
from collections import OrderedDict

__all__ = ["MinimaxAgent"]



class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
    """
    # 自适应深度总开关
    ADAPTIVE_DEPTH_ENABLED = True
    # 分阶段深度（显式可调参数）
    DEPTH_OPENING = 4
    DEPTH_MIDGAME = 4
    DEPTH_ENDGAME = 5
    # 阶段判定阈值（显式可调参数）
    OPENING_OCCUPANCY_THRESHOLD = 0.22
    MIDGAME_OCCUPANCY_THRESHOLD = 0.45
    # 分支收敛后的深度提升参数（显式可调参数）
    LOW_BRANCH_PLAY_COUNT = 8
    LOW_BRANCH_BOOST_DEPTH = 5
    # 当关闭自适应时使用的固定深度
    FIXED_DEPTH = 4
    # 缓存容量（条目数），超出后按 LRU 淘汰
    CACHE_MAX_ENTRIES = 50000

    def __init__(self, max_depth=None, evaluator=None):
        # 中文补充：保留 max_depth 参数仅做兼容；默认以类内显式配置为准
        if max_depth is None:
            self.max_depth = self._configured_max_depth()
        else:
            self.max_depth = self._sanitize_depth(max_depth)
        self.evaluator = evaluator or self._default_evaluator  #评估函数
        self.cache = GameResultCache(max_entries=self.CACHE_MAX_ENTRIES) # 复用局面缓存，减少重复搜索
        self._search_player = Player.black # 记录根节点视角，保证整棵树评估口径一致
        self._active_depth = self.max_depth
        self._cache_hits = 0
        self._cache_misses = 0

    @classmethod
    def set_phase_depths(cls, opening=None, midgame=None, endgame=None):
        # 中文补充：运行时批量调整开中后盘深度
        if opening is not None:
            cls.DEPTH_OPENING = cls._sanitize_depth(opening)
        if midgame is not None:
            cls.DEPTH_MIDGAME = cls._sanitize_depth(midgame)
        if endgame is not None:
            cls.DEPTH_ENDGAME = cls._sanitize_depth(endgame)

    @classmethod
    def set_phase_thresholds(cls, opening=None, midgame=None):
        # 中文补充：运行时调整阶段分界占盘率阈值
        if opening is not None:
            cls.OPENING_OCCUPANCY_THRESHOLD = float(opening)
        if midgame is not None:
            cls.MIDGAME_OCCUPANCY_THRESHOLD = float(midgame)

    @classmethod
    def _configured_max_depth(cls):
        # 中文补充：由显式参数自动推导“配置深度上限”
        if not cls.ADAPTIVE_DEPTH_ENABLED:
            return cls._sanitize_depth(cls.FIXED_DEPTH)
        return max(
            cls._sanitize_depth(cls.DEPTH_OPENING),
            cls._sanitize_depth(cls.DEPTH_MIDGAME),
            cls._sanitize_depth(cls.DEPTH_ENDGAME),
            cls._sanitize_depth(cls.LOW_BRANCH_BOOST_DEPTH),
        )

    @staticmethod
    def _sanitize_depth(depth):
        # 统一深度合法化，避免非法值导致搜索异常
        try:
            depth_int = int(depth)
        except (TypeError, ValueError):
            depth_int = 1
        return max(1, depth_int)

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """

        start_ts = time.time()
        searched_moves = 0 # 统计搜索的棋步数量，评估效率
        active_depth = self.max_depth
        try:
            if game_state.is_over(): # 终局直接过手（或认输），不搜索
                return Move.pass_turn()

            self._search_player = game_state.next_player # 记录搜索视角，确保评估函数一致
            self._active_depth = self._choose_depth(game_state)
            active_depth = self._active_depth
            self._cache_hits = 0
            self._cache_misses = 0
            best_move = Move.pass_turn() # 默认过手，除非找到更好棋步
            best_value = -float("inf") # 初始化为负无穷，确保任何合法棋步都能覆盖

            # 初始化 alpha-beta 边界，最大化方从负无穷开始，最小化方从正无穷开始
            alpha = -float("inf")
            beta = float("inf")

            # 获取排序后的候选棋步，提升剪枝效率
            moves = self._get_ordered_moves(game_state)
            if not moves:
                return best_move

            for move in moves:
                # 默认不主动认输，除非没有其他合法落子（legal_moves 中通常总有 pass）
                if move.is_resign:
                    continue
                searched_moves += 1
                next_state = game_state.apply_move(move)
                value = self.alphabeta(
                    next_state,
                    active_depth - 1,
                    alpha,
                    beta,
                    maximizing_player=False,
                )

                if value > best_value:
                    best_value = value
                    best_move = move

                alpha = max(alpha, best_value)

            return best_move
        finally:
            think_time = time.time() - start_ts
            print(
                "[MINIMAX-TIME] player=%s think_time=%.3fs depth=%d(config=%d) "
                "searched_moves=%d cache_size=%d cache_hits=%d cache_misses=%d"
                % (
                    game_state.next_player.name,
                    think_time,
                    active_depth,
                    self.max_depth,
                    searched_moves,
                    self.cache.size(),
                    self._cache_hits,
                    self._cache_misses,
                )
            )

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            maximizing_player: 是否在当前层最大化（True=我方）

        Returns:
            该局面的评估值
        """
        # TODO: 实现 Minimax
        # 提示：
        # 1. 终局或 depth=0 时返回评估值
        # 2. 如果是最大化方：取所有子节点最大值
        # 3. 如果是最小化方：取所有子节点最小值
        if depth <= 0 or game_state.is_over():
            return self.evaluator(game_state)

        moves = self._get_ordered_moves(game_state)
        if not moves:
            return self.evaluator(game_state)

        if maximizing_player:
            value = -float("inf")
            for move in moves:
                next_state = game_state.apply_move(move)
                value = max(value, self.minimax(next_state, depth - 1, False))
            return value

        value = float("inf")
        for move in moves:
            next_state = game_state.apply_move(move)
            value = min(value, self.minimax(next_state, depth - 1, True))
        return value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax。

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        # 实现 Alpha-Beta 剪枝
        # 提示：在 minimax 基础上添加剪枝逻辑
        # - 最大化方：如果 value >= beta 则剪枝
        # - 最小化方：如果 value <= alpha 则剪枝
        alpha_orig = alpha
        beta_orig = beta

        # 缓存键包含执子方和历史，避免 ko 历史不同导致误命中
        cache_key = (
            game_state.board.zobrist_hash(),
            game_state.next_player,
            self._search_player,
        )
        cached = self.cache.get(cache_key)
        if cached is not None and cached["depth"] >= depth:
            self._cache_hits += 1
            flag = cached["flag"]
            value = cached["value"]
            if flag == "exact":
                return value
            if flag == "lower":
                alpha = max(alpha, value)
            elif flag == "upper":
                beta = min(beta, value)
            if alpha >= beta:
                return value
        else:
            self._cache_misses += 1

        if depth <= 0 or game_state.is_over():
            value = self.evaluator(game_state)
            self.cache.put(cache_key, depth, value, flag="exact")
            return value

        moves = self._get_ordered_moves(game_state)
        if not moves:
            value = self.evaluator(game_state)
            self.cache.put(cache_key, depth, value, flag="exact")
            return value

        if maximizing_player:
            value = -float("inf")
            for move in moves:
                next_state = game_state.apply_move(move)
                value = max(
                    value,
                    self.alphabeta(next_state, depth - 1, alpha, beta, False),
                )
                alpha = max(alpha, value)
                if value >= beta:
                    break
        else:
            value = float("inf")
            for move in moves:
                next_state = game_state.apply_move(move)
                value = min(
                    value,
                    self.alphabeta(next_state, depth - 1, alpha, beta, True),
                )
                beta = min(beta, value)
                if value <= alpha:
                    break

        if value <= alpha_orig:
            flag = "upper"
        elif value >= beta_orig:
            flag = "lower"
        else:
            flag = "exact"
        self.cache.put(cache_key, depth, value, flag=flag)
        return value

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数（简单版本）。

        学生作业：替换为更复杂的评估函数，如：
            - 气数统计
            - 眼位识别
            - 神经网络评估

        Args:
            game_state: 游戏状态

        Returns:
            评估值（正数对我方有利）
        """
        # TODO: 实现简单的启发式评估
        # 示例：子数差 + 气数差
        perspective = self._search_player

        if game_state.is_over():
            winner = game_state.winner()
            if winner is None:
                return 0.0
            return 100000.0 if winner == perspective else -100000.0

        features = self._extract_features(game_state)
        my = perspective
        opp = my.other

        stone_diff = features[my]["stones"] - features[opp]["stones"]
        liberty_diff = features[my]["liberties"] - features[opp]["liberties"]
        capture_diff = (
            features[my]["capture_potential"] - features[opp]["capture_potential"]
        )
        save_diff = features[my]["save_potential"] - features[opp]["save_potential"]
        connection_diff = features[my]["connection"] - features[opp]["connection"]
        center_diff = features[my]["center"] - features[opp]["center"]
        weak_diff = features[opp]["weak_strings"] - features[my]["weak_strings"]
        # 恢复：非终局也使用规则一致的真实计分，避免轻量 proxy 带来的评估偏差
        score_result = compute_game_result(game_state)
        territory_margin = score_result.b - (score_result.w + score_result.komi)
        if my == Player.white:
            territory_margin = -territory_margin

        weights = self._phase_weights(features["total_stones"], features["board_points"]) # 根据棋盘占用率调整权重，开盘更看潜力，后盘更看实地和战术兑现
        """综合多个特征加权评估，模拟人类棋感，提升 Minimax 评估质量"""
        return (
            weights["territory"] * territory_margin
            + weights["stone"] * stone_diff 
            + weights["capture"] * capture_diff
            + weights["save"] * save_diff
            + weights["liberty"] * liberty_diff
            + weights["connection"] * connection_diff
            + weights["center"] * center_diff
            + weights["weak"] * weak_diff
        )

    def _extract_features(self, game_state):
        # 提取 5x5 上对 Minimax 有效且接近 MCTS 直觉的静态特征
        board = game_state.board
        center_row = (board.num_rows + 1) / 2.0
        center_col = (board.num_cols + 1) / 2.0

        features = {
            Player.black: {
                "stones": 0,
                "liberties": 0,
                "weak_strings": 0,
                "capture_potential": 0,
                "save_potential": 0,
                "connection": 0,
                "center": 0.0,
            },
            Player.white: {
                "stones": 0,
                "liberties": 0,
                "weak_strings": 0,
                "capture_potential": 0,
                "save_potential": 0,
                "connection": 0,
                "center": 0.0,
            },
            "total_stones": 0,
            "board_points": board.num_rows * board.num_cols,
        }

        seen_strings = set()

        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                point = Point(row=row, col=col)
                stone = board.get(point)

                if stone is not None:
                    features[stone]["stones"] += 1
                    features["total_stones"] += 1
                    dist = abs(point.row - center_row) + abs(point.col - center_col)
                    features[stone]["center"] += max(0.0, 2.4 - 0.55 * dist)

                    go_string = board.get_go_string(point)
                    if go_string is None:
                        continue
                    string_id = id(go_string)
                    if string_id in seen_strings:
                        continue
                    seen_strings.add(string_id)
                    features[stone]["liberties"] += go_string.num_liberties
                    if go_string.num_liberties == 1:
                        features[stone]["weak_strings"] += 1
                    continue

                # 空点上看战术潜力（打吃/救子/连通）
                neighbor_strings = {}
                adjacent_friend_groups = {
                    Player.black: set(),
                    Player.white: set(),
                }
                for neighbor in point.neighbors():
                    if not board.is_on_grid(neighbor):
                        continue
                    nb_stone = board.get(neighbor)
                    if nb_stone is None:
                        continue
                    nb_string = board.get_go_string(neighbor)
                    if nb_string is None:
                        continue
                    sid = id(nb_string)
                    neighbor_strings[sid] = (nb_stone, nb_string.num_liberties)
                    adjacent_friend_groups[nb_stone].add(sid)

                for color in (Player.black, Player.white):
                    if len(adjacent_friend_groups[color]) >= 2:
                        features[color]["connection"] += 1

                for stone_color, liberties in neighbor_strings.values():
                    if liberties == 1:
                        features[stone_color.other]["capture_potential"] += 1
                        features[stone_color]["save_potential"] += 1

        return features

    def _phase_weights(self, total_stones, board_points):
        # 按开中后盘调整权重，5x5 上收官很快，后盘更看重实地和战术兑现
        occupancy = total_stones / float(max(1, board_points))
        if occupancy < 0.35:
            return {
                "territory": 1.8,
                "stone": 0.7,
                "capture": 1.4,
                "save": 0.9,
                "liberty": 0.7,
                "connection": 1.0,
                "center": 0.8,
                "weak": 1.1,
            }
        if occupancy < 0.72:
            return {
                "territory": 2.4,
                "stone": 0.6,
                "capture": 1.9,
                "save": 1.1,
                "liberty": 0.9,
                "connection": 0.8,
                "center": 0.4,
                "weak": 1.4,
            }
        return {
            "territory": 3.2,
            "stone": 0.4,
            "capture": 2.2,
            "save": 1.2,
            "liberty": 0.6,
            "connection": 0.3,
            "center": 0.1,
            "weak": 1.6,
        }

    def _choose_depth(self, game_state):
        # 自适应深度：开局降深度控时，中后盘升深度提强度
        max_depth = self.max_depth
        if not self.ADAPTIVE_DEPTH_ENABLED:
            return self._sanitize_depth(min(max_depth, self.FIXED_DEPTH))

        board = game_state.board
        board_points = board.num_rows * board.num_cols
        total_stones = 0
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                if board.get(Point(row, col)) is not None:
                    total_stones += 1
        occupancy = total_stones / float(max(1, board_points))
        legal_play_count = sum(1 for m in game_state.legal_moves() if m.is_play)

        opening_depth = self._sanitize_depth(self.DEPTH_OPENING)
        midgame_depth = self._sanitize_depth(self.DEPTH_MIDGAME)
        endgame_depth = self._sanitize_depth(self.DEPTH_ENDGAME)

        if occupancy < self.OPENING_OCCUPANCY_THRESHOLD:
            depth = opening_depth
        elif occupancy < self.MIDGAME_OCCUPANCY_THRESHOLD:
            depth = midgame_depth
        else:
            depth = endgame_depth

        # 分支数已经显著下降时，允许提深度
        if legal_play_count <= self.LOW_BRANCH_PLAY_COUNT:
            boost_depth = self._sanitize_depth(self.LOW_BRANCH_BOOST_DEPTH)
            depth = max(depth, min(boost_depth, max_depth))
        return min(max_depth, max(1, depth))

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """

        moves = game_state.legal_moves()
        play_moves = []
        pass_move = None
        resign_move = None

        for move in moves:
            if move.is_play:
                play_moves.append(move)
            elif move.is_pass:
                pass_move = move
            elif move.is_resign:
                resign_move = move

        board = game_state.board
        current_player = game_state.next_player
        scored_moves = []

        # 按“吃子/救子/连络/气数/中心”综合排序，提升剪枝效率
        for move in play_moves:
            point = move.point
            score = 0.0

            center_row = (board.num_rows + 1) / 2.0
            center_col = (board.num_cols + 1) / 2.0
            dist_to_center = abs(point.row - center_row) + abs(point.col - center_col)
            score += 2.0 - 0.30 * dist_to_center

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

            next_state = game_state.apply_move(move)
            new_string = next_state.board.get_go_string(point)
            new_liberties = new_string.num_liberties if new_string is not None else 0

            score += 0.35 * friendly_neighbors
            score += 0.45 * enemy_neighbors
            score += 3.2 * capture_targets
            score += 2.0 * save_targets
            score += 0.40 * new_liberties

            scored_moves.append((score, move))

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        ordered = [move for _, move in scored_moves]

        # pass 作为兜底候选，resign 始终放在最后
        if pass_move is not None:
            ordered.append(pass_move)
        if resign_move is not None:
            ordered.append(resign_move)

        return ordered  # 目前无序



class GameResultCache:
    """
    局面缓存（Transposition Table）。

    用 Zobrist 哈希缓存已评估的局面，避免重复计算。
    """

    def __init__(self, max_entries=50000):
        self.max_entries = max(1000, int(max_entries))
        self.cache = OrderedDict()

    def get(self, zobrist_hash):
        """获取缓存的评估值。"""
        item = self.cache.get(zobrist_hash)
        if item is None:
            return None
        # LRU: 最近命中的条目放到队尾
        self.cache.move_to_end(zobrist_hash)
        return item

    def put(self, zobrist_hash, depth, value, flag='exact'):
        """
        缓存评估结果。

        Args:
            zobrist_hash: 局面哈希
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        old = self.cache.get(zobrist_hash)
        # 仅当新条目搜索深度更深（或同深）时覆盖旧值
        if old is None or depth >= old["depth"]:
            self.cache[zobrist_hash] = {
                "depth": depth,
                "value": value,
                "flag": flag,
            }
            self.cache.move_to_end(zobrist_hash)
        if len(self.cache) > self.max_entries:
            # LRU: 淘汰最久未访问条目
            self.cache.popitem(last=False)

    def size(self):
        return len(self.cache)
