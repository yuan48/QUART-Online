import numpy as np

# 假设你的文件名为 'data.npy'
crawl_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data/vq_Dataset_crawl_red_gate1.npy'
tunnel_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data/vq_Dataset_go_through_green_rectangle tunnel.npy'
goto_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data/vq_Dataset_go_to_green_cube.npy'
unload_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data/vq_Dataset_unload_green_traybox.npy'
avoid_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data/vq_Dataset_go_avoid_green_cube.npy'
# file_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data/vq_Dataset_crawl_red_gate1.npy'

# 使用 numpy 的 load 函数读取 .npy 文件
crawl = np.load(crawl_path,allow_pickle=True)
tunnel = np.load(tunnel_path,allow_pickle=True)
goto = np.load(goto_path,allow_pickle=True)
unload = np.load(unload_path,allow_pickle=True)
avoid = np.load(avoid_path,allow_pickle=True)

# 打印读取的数据
# print(data)
import pdb; pdb.set_trace()