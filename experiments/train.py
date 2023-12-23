import train_utils as train_utils
import os
# nohup python -u experiments/train.py --log_dir results/point_goal2_0limit_seed30/ --environment sgym_Safexp-PointGoal2-v0 --total_training_steps 5000000 --safety --cuda_device 7 --seed 30 --cost_threshold 2.0 > pointgoal2_0limit_seed30.log 2>&1 &
# python experiments/train.py --log_dir results/test_1223/ --environment sgym_Safexp-PointGoal2-v0 --total_training_steps 5000000 --safety --cuda_device 7 --seed 30 --cost_threshold 2.0
if __name__ == '__main__':
    config = train_utils.make_config(train_utils.define_config())
    print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_device

    from la_mbda.la_mbda import LAMBDA

    train_utils.train(config, LAMBDA)
