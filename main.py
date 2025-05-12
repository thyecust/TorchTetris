import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import random

# --- 游戏配置 ---
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
GRAVITY_INTERVAL_INITIAL = 0.8  # 秒，方块自动下落的初始间隔
GRAVITY_INTERVAL_MIN = 0.1    # 秒，最快下落间隔
LEVEL_UP_LINES = 10           # 每消除多少行升一级（速度加快）

# --- 方块定义 (0和1表示形状, id用于颜色和区分) ---
# 形状的定义：列表中的每个元素是一个 torch.tensor 代表一种旋转形态
_I = [torch.tensor([[1,1,1,1]], dtype=torch.int), torch.tensor([[1],[1],[1],[1]], dtype=torch.int)]
_J = [torch.tensor([[1,0,0],[1,1,1]], dtype=torch.int), torch.tensor([[0,1],[0,1],[1,1]], dtype=torch.int), torch.tensor([[1,1,1],[0,0,1]], dtype=torch.int), torch.tensor([[1,1],[1,0],[1,0]], dtype=torch.int)]
_L = [torch.tensor([[0,0,1],[1,1,1]], dtype=torch.int), torch.tensor([[1,1],[0,1],[0,1]], dtype=torch.int), torch.tensor([[1,1,1],[1,0,0]], dtype=torch.int), torch.tensor([[1,0],[1,0],[1,1]], dtype=torch.int)]
_O = [torch.tensor([[1,1],[1,1]], dtype=torch.int)]
_S = [torch.tensor([[0,1,1],[1,1,0]], dtype=torch.int), torch.tensor([[1,0],[1,1],[0,1]], dtype=torch.int)]
_Z = [torch.tensor([[1,1,0],[0,1,1]], dtype=torch.int), torch.tensor([[0,1],[1,1],[1,0]], dtype=torch.int)]
_T = [torch.tensor([[0,1,0],[1,1,1]], dtype=torch.int), torch.tensor([[1,0],[1,1],[1,0]], dtype=torch.int), torch.tensor([[1,1,1],[0,1,0]], dtype=torch.int), torch.tensor([[0,1],[1,1],[0,1]], dtype=torch.int)]

TETROMINOES_DEF = [
    {'id': 1, 'shapes': _I, 'name': 'I'},
    {'id': 2, 'shapes': _J, 'name': 'J'},
    {'id': 3, 'shapes': _L, 'name': 'L'},
    {'id': 4, 'shapes': _O, 'name': 'O'},
    {'id': 5, 'shapes': _S, 'name': 'S'},
    {'id': 6, 'shapes': _Z, 'name': 'Z'},
    {'id': 7, 'shapes': _T, 'name': 'T'},
]

# --- 颜色映射 ---
# 0: 背景 (黑色), 1-7: 不同方块的颜色
COLORS = ['black', 'cyan', 'blue', 'orange', 'yellow', 'lime', 'purple', 'red']
CMAP = mcolors.ListedColormap(COLORS)
NORM = mcolors.BoundaryNorm(list(range(len(COLORS) + 1)), CMAP.N)

# --- 游戏状态初始化 ---
def initialize_game_state():
    """初始化游戏状态字典"""
    state = {
        'board': torch.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.int),
        'current_piece_data': None, # 将由 spawn_new_piece 填充
        'score': 0,
        'lines_cleared_total': 0,
        'level': 1,
        'gravity_interval': GRAVITY_INTERVAL_INITIAL,
        'game_over': False,
        'pending_action': None, # 存储用户输入
    }
    spawn_new_piece(state) # 生成第一个方块
    return state

# --- 核心游戏逻辑函数 ---
def spawn_new_piece(state):
    """在游戏板顶部生成一个新的随机方块"""
    if state['game_over']:
        return

    piece_idx = random.randint(0, len(TETROMINOES_DEF) - 1)
    piece_def = TETROMINOES_DEF[piece_idx]
    rotation_idx = random.randint(0, len(piece_def['shapes']) - 1) # 随机初始旋转
    shape_tensor = piece_def['shapes'][rotation_idx] # 这是0/1的模板
    piece_id = piece_def['id']

    # 计算初始位置 (顶部居中)
    pos_c = (BOARD_WIDTH - shape_tensor.shape[1]) // 2
    pos_r = 0 

    state['current_piece_data'] = {
        'type_idx': piece_idx,
        'rotation_idx': rotation_idx,
        'tensor': shape_tensor, # 存储0/1形状模板
        'id': piece_id,         # 方块的ID，用于着色和在板上标记
        'pos': [pos_r, pos_c]   # [行, 列] 方块左上角在板上的位置
    }

    # 检查新生成的方块是否立即导致碰撞 (游戏结束条件)
    if not is_valid_move(state['board'], shape_tensor, [pos_r, pos_c]):
        state['game_over'] = True
        # 为了确保游戏结束时当前方块能显示出来，尝试将其放置
        # 这可能覆盖现有方块，但在游戏结束时视觉上更清晰
        if state['current_piece_data']:
             place_piece(state['board'], 
                        state['current_piece_data']['tensor'], 
                        state['current_piece_data']['pos'], 
                        state['current_piece_data']['id'],
                        force_place=True)


def is_valid_move(board, piece_tensor, piece_pos):
    """检查将 piece_tensor 放置在 board 的 piece_pos 位置是否有效"""
    piece_height, piece_width = piece_tensor.shape
    pos_r, pos_c = piece_pos

    for r_idx in range(piece_height):
        for c_idx in range(piece_width):
            if piece_tensor[r_idx, c_idx] > 0: # 如果这是方块的一部分
                board_r, board_c = pos_r + r_idx, pos_c + c_idx
                
                # 检查是否越界
                if not (0 <= board_r < BOARD_HEIGHT and 0 <= board_c < BOARD_WIDTH):
                    return False
                # 检查是否与板上已有的方块碰撞
                if board[board_r, board_c] > 0:
                    return False
    return True

def place_piece(board, piece_tensor, piece_pos, piece_id, force_place=False):
    """将 piece_tensor (用 piece_id 标记) 放置到 board 的 piece_pos 位置"""
    piece_height, piece_width = piece_tensor.shape
    pos_r, pos_c = piece_pos
    for r_idx in range(piece_height):
        for c_idx in range(piece_width):
            if piece_tensor[r_idx, c_idx] > 0:
                board_r, board_c = pos_r + r_idx, pos_c + c_idx
                # 确保只在板内放置，除非是强制放置（如游戏结束时）
                if force_place or (0 <= board_r < BOARD_HEIGHT and 0 <= board_c < BOARD_WIDTH) :
                     # 处理边界情况，避免在force_place时越界写入
                    if 0 <= board_r < BOARD_HEIGHT and 0 <= board_c < BOARD_WIDTH:
                        board[board_r, board_c] = piece_id


def clear_lines(state):
    """检查并清除 board 上的满行，更新分数和等级"""
    board = state['board']
    lines_cleared_this_turn = 0
    write_row = BOARD_HEIGHT - 1 # 从底部开始检查和写入

    for read_row in range(BOARD_HEIGHT - 1, -1, -1): # 从下往上读
        if not (board[read_row, :] > 0).all(): # 如果当前行未满
            if write_row != read_row: # 如果需要移动
                board[write_row, :] = board[read_row, :]
            write_row -= 1
        else: # 当前行为满行
            lines_cleared_this_turn += 1
            state['lines_cleared_total'] += 1

    # 用空行填充顶部
    while write_row >= 0:
        board[write_row, :] = 0
        write_row -= 1

    if lines_cleared_this_turn > 0:
        # 计分逻辑 (可以更复杂，例如根据一次消除的行数给予不同分数)
        score_map = {1: 40, 2: 100, 3: 300, 4: 1200} # 单行，双行，三行，俄罗斯方块
        state['score'] += score_map.get(lines_cleared_this_turn, 0) * state['level']
        
        # 检查是否升级
        if state['lines_cleared_total'] // LEVEL_UP_LINES >= state['level']:
            state['level'] +=1
            state['gravity_interval'] = max(GRAVITY_INTERVAL_MIN, GRAVITY_INTERVAL_INITIAL - (state['level'] -1) * 0.05)
            print(f"Level Up! Level: {state['level']}, Speed: {state['gravity_interval']:.2f}s")

    return lines_cleared_this_turn


def handle_action(state, action):
    """处理用户输入或游戏事件 (如 'left', 'right', 'rotate', 'down', 'drop')"""
    if state['game_over'] or not state['current_piece_data']:
        return

    piece_data = state['current_piece_data']
    board = state['board']
    current_pos = list(piece_data['pos']) # 创建副本以进行修改
    current_shape_tensor = piece_data['tensor']

    if action == 'left':
        new_pos = [current_pos[0], current_pos[1] - 1]
        if is_valid_move(board, current_shape_tensor, new_pos):
            piece_data['pos'] = new_pos
    elif action == 'right':
        new_pos = [current_pos[0], current_pos[1] + 1]
        if is_valid_move(board, current_shape_tensor, new_pos):
            piece_data['pos'] = new_pos
    elif action == 'down': # 软降
        new_pos = [current_pos[0] + 1, current_pos[1]]
        if is_valid_move(board, current_shape_tensor, new_pos):
            piece_data['pos'] = new_pos
        else: # 如果软降导致碰撞，则锁定方块
            apply_gravity_and_lock(state, force_lock=True)
    elif action == 'rotate':
        piece_def = TETROMINOES_DEF[piece_data['type_idx']]
        current_rotation_idx = piece_data['rotation_idx']
        next_rotation_idx = (current_rotation_idx + 1) % len(piece_def['shapes'])
        next_shape_tensor = piece_def['shapes'][next_rotation_idx]
        
        # 简单的旋转尝试，没有复杂的墙体反弹逻辑
        # 可以尝试一些基本偏移（墙体反弹）
        # offsets = [[0,0], [0,-1], [0,1], [-1,0], [1,0]] # (dr, dc)
        # if piece_def['name'] == 'I': # I型方块的偏移可能更大
        #     offsets.extend([[0,-2], [0,2]])

        # 简化：仅在当前位置尝试旋转
        if is_valid_move(board, next_shape_tensor, current_pos):
            piece_data['tensor'] = next_shape_tensor
            piece_data['rotation_idx'] = next_rotation_idx
        # else: # 尝试墙体反弹 (简化版)
            # for dr, dc in offsets:
            #     if dr == 0 and dc == 0: continue # 跳过原始位置
            #     kick_pos = [current_pos[0] + dr, current_pos[1] + dc]
            #     if is_valid_move(board, next_shape_tensor, kick_pos):
            #         piece_data['tensor'] = next_shape_tensor
            #         piece_data['rotation_idx'] = next_rotation_idx
            #         piece_data['pos'] = kick_pos
            #         break


    elif action == 'drop': # 硬降
        temp_pos = list(current_pos)
        while True:
            next_temp_pos = [temp_pos[0] + 1, temp_pos[1]]
            if is_valid_move(board, current_shape_tensor, next_temp_pos):
                temp_pos = next_temp_pos
            else:
                break
        piece_data['pos'] = temp_pos
        apply_gravity_and_lock(state, force_lock=True) # 硬降后立即锁定

def apply_gravity_and_lock(state, force_lock=False):
    """应用重力使方块下落一格，如果碰撞则锁定方块，检查消行并生成新方块"""
    if state['game_over'] or not state['current_piece_data']:
        return

    piece_data = state['current_piece_data']
    board = state['board']
    current_pos = piece_data['pos']
    current_shape_tensor = piece_data['tensor']

    new_pos_down = [current_pos[0] + 1, current_pos[1]]

    if not force_lock and is_valid_move(board, current_shape_tensor, new_pos_down):
        piece_data['pos'] = new_pos_down
    else: # 不能再向下移动或被强制锁定
        place_piece(board, current_shape_tensor, current_pos, piece_data['id'])
        clear_lines(state) # 清除满行并更新分数/等级
        spawn_new_piece(state) # 生成新方块 (如果游戏未结束)
        # spawn_new_piece 内部会检查游戏是否结束


def get_renderable_board(state):
    """获取用于渲染的游戏板视图 (包括当前活动方块)"""
    if state['game_over'] and not state['current_piece_data']: # 游戏结束且没有当前方块（可能发生在生成时就碰撞）
        return state['board'].clone()

    render_board = state['board'].clone()
    if state['current_piece_data']: # 确保有当前方块
        piece_data = state['current_piece_data']
        piece_tensor = piece_data['tensor']
        piece_id = piece_data['id']
        pos_r, pos_c = piece_data['pos']
        piece_h, piece_w = piece_tensor.shape

        for r_idx in range(piece_h):
            for c_idx in range(piece_w):
                if piece_tensor[r_idx, c_idx] > 0:
                    render_r, render_c = pos_r + r_idx, pos_c + c_idx
                    if 0 <= render_r < BOARD_HEIGHT and 0 <= render_c < BOARD_WIDTH:
                        render_board[render_r, render_c] = piece_id
    return render_board

# --- Matplotlib 事件处理 ---
def on_key_press(event, game_state_ref):
    """处理键盘按下事件，更新 pending_action"""
    if game_state_ref['game_over']:
        return

    action = None
    if event.key == 'left':
        action = 'left'
    elif event.key == 'right':
        action = 'right'
    elif event.key == 'down':
        action = 'down' # 软降
    elif event.key == 'up' or event.key == 'x': # 'x' 通常用于旋转
        action = 'rotate'
    elif event.key == 'space' or event.key == 'z': # 'z' 也常用于旋转，这里用作硬降
        action = 'drop'   # 硬降
    
    if action:
        game_state_ref['pending_action'] = action


# --- 主游戏循环 ---
if __name__ == "__main__":
    game_state = initialize_game_state()

    plt.ion() # 开启交互模式
    fig, ax = plt.subplots()
    # 初始化图像对象，使用自定义颜色映射和归一化
    img = ax.imshow(torch.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.int), cmap=CMAP, norm=NORM)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.manager.set_window_title('俄罗斯方块 (PyTorch)')


    # 连接键盘事件处理器
    # 使用 lambda 将 game_state 传递给事件处理器
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, game_state))

    last_gravity_time = time.time()

    while not game_state['game_over']:
        current_time = time.time()

        # 1. 处理用户输入
        if game_state['pending_action']:
            handle_action(game_state, game_state['pending_action'])
            game_state['pending_action'] = None # 消耗掉动作

        # 2. 应用重力
        if current_time - last_gravity_time >= game_state['gravity_interval']:
            apply_gravity_and_lock(game_state) # 重力下落并检查锁定
            last_gravity_time = current_time
            if game_state['game_over']: # apply_gravity_and_lock 可能会导致游戏结束
                break
        
        # 3. 渲染游戏板
        display_board = get_renderable_board(game_state)
        img.set_data(display_board.numpy()) # imshow 需要 numpy array
        
        # 更新标题显示分数和等级
        ax.set_title(f"Score: {game_state['score']} | Cleared Lines: {game_state['lines_cleared_total']} | Level: {game_state['level']}")
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events() # 处理UI事件
        
        plt.pause(0.01) # 短暂暂停，保持响应性

    # 游戏结束后的处理
    print(f"Game End! Score: {game_state['score']}, Cleared Lines: {game_state['lines_cleared_total']}, Level: {game_state['level']}")
    ax.set_title(f"Game End! Score: {game_state['score']}, Cleared Lines: {game_state['lines_cleared_total']}, Level: {game_state['level']}", color='red')
    # 确保最后的游戏板状态被渲染
    final_board_view = get_renderable_board(game_state)
    img.set_data(final_board_view.numpy())
    fig.canvas.draw_idle()
    
    plt.ioff() # 关闭交互模式
    plt.show() # 保持窗口打开直到用户关闭
