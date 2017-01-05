import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
#from rllab.envs.normalized_env import NormalizedEnv

from sandbox.vime.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
import argparse

stub(globals())

parser = argparse.ArgumentParser(description='Run trpo algorithm')
parser.add_argument('environment', metavar='env',
        help='environment to run on')
args = parser.parse_args()

# Param ranges
seeds = range(2)
etas = [0.0001]
# SwimmerGather hierarchical task
env = GymEnv(args.environment)
mdp_classes = [env]
#mdp_classes = [SwimmerGatherEnv]
mdps = [normalize(mdp_class)
        for mdp_class in mdp_classes]

param_cart_product = itertools.product(
    mdps, etas, seeds
)

for mdp, eta, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = 50000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=10000,
        step_size=0.01,
        eta=eta,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_replay_pool=True,
        use_kl_ratio=True,
        use_kl_ratio_q=True,
        n_itr_update=1,
        kl_batch_size=1,
        normalize_reward=False,
        replay_pool_size=1000000,
        n_updates_per_sample=5000,
        second_order_update=True,
        #unn_n_hidden=[32],
        unn_n_hidden=[100],
        unn_layers_type=[1, 1],
        unn_learning_rate=0.0001
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-expl",
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        script="sandbox/vime/experiments/run_experiment_lite.py",
    )
