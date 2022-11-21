import argparse

parser = argparse.ArgumentParser()
#Network structure parameter
parser.add_argument('--net_share', type=int, default=16)
parser.add_argument('--net_state', type=int, default=16)
parser.add_argument('--net_action', type=int, default=64)


#Network training parameter
parser.add_argument('--loss', default='mse')
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--capacity', type=int, default=1000)

parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9)
parser.add_argument('--eps_1', type=float, default=0.8)
parser.add_argument('--eps_2', type=float, default=0.4)
parser.add_argument('--eps_min', type=float, default=0.1)

parser.add_argument('--gamma', type=float, default=0.98)

parser.add_argument('--replace_target_iter', type=int, default=1)

parser.add_argument('--save_path', default='Modle/result.h5')
parser.add_argument('--node_info', default='config/sim_node.txt')
parser.add_argument('--line_info', default='config/match_node.txt')
parser.add_argument('--path_info', default='Config/global_path.txt')
parser.add_argument('--send_path', default='Config/send_result.txt')

args = parser.parse_args()