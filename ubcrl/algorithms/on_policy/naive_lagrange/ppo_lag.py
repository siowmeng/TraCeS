from typing import List
import pathlib
import time
import math
import numpy as np
import os
import pandas as pd
import torch as th
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import track

from ubcrl.adapter.onpolicy_adapter import OnPolicyLearnedBCAdapter, OnPolicyLearnedHAdapter, OnPolicyLearnedHumanAdapter, OnPolicyLearnedBCHumanAdapter
from ubcrl.algorithms import registry
from ubcrl.classify.classifier import MujocoNPDataset, HumanNPDataset
from ubcrl.common.buffer import VectorOnPolicyBufferH
from ubcrl.models.actor_critic import ConstraintActorCriticH
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.utils import distributed
from omnisafe.utils.config import Config
from ubcrl.classify.classifier import classifier_nw_class, TrajDFData, collate, collate_maxlength, DistributionGRU, PtEstGRU
from ubcrl.common.lagrange import LagrangeH
import ubcrl.common.utils as utils

import cv2
import imageio


def _validate_perf(dataloader, classifier, running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn):

    for j, valid_data in enumerate(dataloader, 0):

        inputs_valid, labels_valid, input_lengths_valid = valid_data
        labels_valid = labels_valid.reshape(-1, 1)

        valid_loss, num_valid_correct, num_valid_tp, num_valid_fp, num_valid_tn, num_valid_fn = (
            classifier.forward_loss_metrics(inputs_valid, labels_valid.float(), input_lengths_valid, classweight=1.0))

        running_valid_loss += valid_loss.item()

        valid_correct += num_valid_correct
        valid_tp += num_valid_tp
        valid_fp += num_valid_fp
        valid_tn += num_valid_tn
        valid_fn += num_valid_fn

    return running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn


def _calculate_metrics(running_loss: float, correct: int, tp: int, fp: int, tn: int, fn: int, num_batch: int,
                       num_data: int) -> tuple[float, float, float, float]:

    ave_loss = running_loss / num_batch
    accuracy = correct / num_data
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return ave_loss, accuracy, precision, recall


def _retrain_classifier(env_id: str, trajectories: List[pd.DataFrame], classifier: nn.Module,
                        classifier_trainset: MujocoNPDataset, classifier_testset: MujocoNPDataset,
                        # classifier_new_trainset: MujocoNPDataset, classifier_new_testset: MujocoNPDataset,
                        classifier_optimizer: Optimizer, logger: Logger, configs: Config, max_epoch: int,
                        target_acc: float, noise: float = 0.0) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:

    assert max_epoch > 0

    trajectories_data = TrajDFData(trajectories, env_id)
    all_idx = np.arange(trajectories_data.get_num_traj())
    np.random.shuffle(all_idx)
    split_idx = math.ceil(len(all_idx) * 0.1)
    train_idx, test_idx = all_idx[split_idx:], all_idx[:split_idx]

    new_train_dataset = MujocoNPDataset(mujoco_domain=classifier_trainset.domain,
                                        np_data=trajectories_data, indices=all_idx, noise=noise)
                                        # np_data=trajectories_data, indices=train_idx)
    new_train_dataloader = DataLoader(new_train_dataset, batch_size=32, collate_fn=collate_maxlength(trajectories_data.horizon), shuffle=True)
    new_test_dataset = MujocoNPDataset(mujoco_domain=classifier_testset.domain,
                                       np_data=trajectories_data, indices=all_idx, noise=noise)
                                       # np_data=trajectories_data, indices=test_idx)
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=32, collate_fn=collate_maxlength(trajectories_data.horizon), shuffle=True)

    total_queries = new_train_dataset.n_queries + new_test_dataset.n_queries

    # Unfreeze classifier param
    for param in classifier.parameters():
        param.requires_grad_(True)

    classifier.train()
    th.backends.cudnn.enabled = True

    # Retrain classifier for 10 epochs
    epoch = 0
    ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = float('Inf'), float('Inf'), float('-Inf'), float('-Inf'), float('-Inf')
    new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = float('Inf'), float('Inf'), float('-Inf'), float('-Inf')

    if len(classifier_trainset) == 0:
        old_train_dataloader = DataLoader(new_train_dataset, batch_size=64,
                                          collate_fn=collate_maxlength(trajectories_data.horizon),
                                          shuffle=True)
        old_train_dataloader_iter = iter(old_train_dataloader)
        old_test_dataloader = DataLoader(new_test_dataset, batch_size=64,
                                         collate_fn=collate_maxlength(trajectories_data.horizon),
                                         shuffle=True)
    else:
        old_train_dataloader = DataLoader(classifier_trainset, batch_size=64,
                                          collate_fn=collate_maxlength(trajectories_data.horizon),
                                          shuffle=True)
        old_train_dataloader_iter = iter(old_train_dataloader)
        old_test_dataloader = DataLoader(classifier_testset, batch_size=64, collate_fn=collate_maxlength(trajectories_data.horizon),
                                         shuffle=True)

    for idx in track(range(max_epoch), description='Updating Classifier...'):
        classifier.train()
        running_loss, running_loss_train = 0.0, 0.0

        for i, data in enumerate(new_train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            new_inputs, new_labels, new_input_lengths = data
            new_labels = new_labels.reshape(-1, 1)

            try:
                old_data = next(old_train_dataloader_iter)
            except StopIteration:
                if len(classifier_trainset) == 0:
                    old_train_dataloader = DataLoader(new_train_dataset, batch_size=64,
                                                      collate_fn=collate_maxlength(trajectories_data.horizon),
                                                      shuffle=True)
                else:
                    old_train_dataloader = DataLoader(classifier_trainset, batch_size=64,
                                                      collate_fn=collate_maxlength(trajectories_data.horizon),
                                                      shuffle=True)
                old_train_dataloader_iter = iter(old_train_dataloader)
                old_data = next(old_train_dataloader_iter)

            # old_inputs, old_labels, old_input_lengths = next(old_train_dataloader_iter)
            old_inputs, old_labels, old_input_lengths = old_data
            old_labels = old_labels.reshape(-1, 1)

            inputs = th.cat([new_inputs, old_inputs])
            labels = th.cat([new_labels, old_labels])
            input_lengths = th.cat([new_input_lengths, old_input_lengths])

            loss, num_correct, num_tp, num_fp, num_tn, num_fn = (
                classifier.forward_loss_metrics(inputs, labels.float(), input_lengths, classweight=1.0))

            # zero the parameter gradients
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_train += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        ave_training_loss = running_loss_train / (len(new_train_dataloader) * 2)

        print('[%d] Training loss: %.4f' % (epoch + 1, ave_training_loss))

        classifier.eval()

        running_valid_loss, valid_correct = 0.0, 0
        valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0

        running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn =(
            _validate_perf(new_test_dataloader, classifier, running_valid_loss,
                           valid_correct, valid_tp, valid_fp, valid_tn, valid_fn)
        )
        new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = (
            _calculate_metrics(running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn,
                               len(new_test_dataloader),
                               len(new_test_dataloader.dataset))
        )

        print('[%d] Validation loss (New Trajectories): %.4f' % (epoch + 1, new_valid_loss))
        print(
            f"Validation Error (New Trajectories): \n Accuracy: {(100 * new_valid_accuracy):>0.1f}%, Avg loss: {new_valid_loss:>8f} \n")
        print(f"Validation Error (New Trajectories): \n Precision: {(100 * new_valid_precision):>0.1f}% \n")
        print(f"Validation Error (New Trajectories): \n Recall: {(100 * new_valid_recall):>0.1f}% \n")

        running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn = (
            _validate_perf(old_test_dataloader, classifier, running_valid_loss,
                           valid_correct, valid_tp, valid_fp, valid_tn, valid_fn)
        )
        valid_loss, valid_accuracy, valid_precision, valid_recall = (
            _calculate_metrics(running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn,
                               len(new_test_dataloader) + len(old_test_dataloader),
                               len(new_test_dataloader.dataset) + len(old_test_dataloader.dataset))
        )

        print('[%d] Validation loss (All Trajectories): %.4f' % (epoch + 1, valid_loss))
        print(
            f"Validation Error (All Trajectories): \n Accuracy: {(100 * valid_accuracy):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
        print(f"Validation Error (All Trajectories): \n Precision: {(100 * valid_precision):>0.1f}% \n")
        print(f"Validation Error (All Trajectories): \n Recall: {(100 * valid_recall):>0.1f}% \n")
        epoch += 1

        # if valid_accuracy >= 0.95:
        # if valid_accuracy >= target_acc:
        # if valid_accuracy >= 0.99:
        if new_valid_accuracy >= target_acc:
            logger.log(f'Early stopping at iter {idx + 1} due to desired accuracy reached')
            break

    # Freeze classifier param
    for param in classifier.parameters():
        param.requires_grad_(False)

    classifier.eval()
    th.backends.cudnn.enabled = False

    classifier_trainset.add_augment_data(np_data=trajectories_data, indices=train_idx, noise=noise)
    classifier_testset.add_augment_data(np_data=trajectories_data, indices=test_idx, noise=noise)

    # if configs.model_cfgs.classifier.save_dir is not None:
    #     pathlib.Path(configs.model_cfgs.classifier.save_dir).mkdir(parents=True, exist_ok=True)
    #     th.save(classifier.state_dict(), os.path.join(configs.model_cfgs.classifier.save_dir,
    #                                                   'Classifier-' + str(epoch) + '.pt'))

    # Add to current Logger
    return (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
            (new_valid_precision, valid_precision), (new_valid_recall, valid_recall), total_queries
            )


# def _retrain_classifier(env_id: str, trajectories: List[pd.DataFrame], classifier: nn.Module,
#                         classifier_trainset: MujocoNPDataset, classifier_testset: MujocoNPDataset,
#                         # classifier_new_trainset: MujocoNPDataset, classifier_new_testset: MujocoNPDataset,
#                         classifier_optimizer: Optimizer, logger: Logger, configs: Config, max_epoch: int,
#                         target_acc: float) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:
#
#     assert max_epoch > 0
#
#     trajectories_data = TrajDFData(trajectories, env_id)
#     all_idx = np.arange(trajectories_data.get_num_traj())
#     np.random.shuffle(all_idx)
#     split_idx = math.ceil(len(all_idx) * 0.1)
#     train_idx, test_idx = all_idx[split_idx:], all_idx[:split_idx]
#
#     classifier_trainset.add_augment_data(np_data=trajectories_data, indices=all_idx)
#     classifier_testset.add_augment_data(np_data=trajectories_data, indices=all_idx)
#
#     new_train_dataset = MujocoNPDataset(mujoco_domain=classifier_trainset.domain,
#                                         np_data=trajectories_data, indices=all_idx)
#                                         # np_data=trajectories_data, indices=train_idx)
#     new_train_dataloader = DataLoader(new_train_dataset, batch_size=64, collate_fn=collate_maxlength(trajectories_data.horizon), shuffle=True)
#     new_test_dataset = MujocoNPDataset(mujoco_domain=classifier_testset.domain,
#                                        np_data=trajectories_data, indices=all_idx)
#                                        # np_data=trajectories_data, indices=test_idx)
#     new_test_dataloader = DataLoader(new_test_dataset, batch_size=64, collate_fn=collate_maxlength(trajectories_data.horizon), shuffle=True)
#
#     total_queries = new_train_dataset.n_queries
#
#     # Unfreeze classifier param
#     for param in classifier.parameters():
#         param.requires_grad_(True)
#
#     classifier.train()
#     th.backends.cudnn.enabled = True
#
#     # Retrain classifier for 10 epochs
#     epoch = 0
#     ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = float('Inf'), float('Inf'), float('-Inf'), float('-Inf'), float('-Inf')
#     new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = float('Inf'), float('Inf'), float('-Inf'), float('-Inf')
#
#     old_train_dataloader = DataLoader(classifier_trainset, batch_size=64,
#                                       collate_fn=collate_maxlength(trajectories_data.horizon),
#                                       shuffle=True)
#     old_train_dataloader_iter = iter(old_train_dataloader)
#     old_test_dataloader = DataLoader(classifier_testset, batch_size=64, collate_fn=collate_maxlength(trajectories_data.horizon),
#                                      shuffle=True)
#
#     for idx in track(range(max_epoch), description='Updating Classifier...'):
#         classifier.train()
#         running_loss, running_loss_train = 0.0, 0.0
#
#         for i, data in enumerate(old_train_dataloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels, input_lengths = data
#             labels = labels.reshape(-1, 1)
#
#             loss, num_correct, num_tp, num_fp, num_tn, num_fn = (
#                 classifier.forward_loss_metrics(inputs, labels.float(), input_lengths, classweight=1.0))
#
#             # zero the parameter gradients
#             classifier_optimizer.zero_grad()
#             loss.backward()
#             classifier_optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             running_loss_train += loss.item()
#             if i % 100 == 99:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
#                 running_loss = 0.0
#
#         ave_training_loss = running_loss_train / (len(old_train_dataloader) * 2)
#
#         print('[%d] Training loss: %.4f' % (epoch + 1, ave_training_loss))
#
#         classifier.eval()
#
#         running_valid_loss, valid_correct = 0.0, 0
#         valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0
#
#         running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn =(
#             _validate_perf(new_test_dataloader, classifier, running_valid_loss,
#                            valid_correct, valid_tp, valid_fp, valid_tn, valid_fn)
#         )
#         new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = (
#             _calculate_metrics(running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn,
#                                len(new_test_dataloader),
#                                len(new_test_dataloader.dataset))
#         )
#
#         print('[%d] Validation loss (New Trajectories): %.4f' % (epoch + 1, new_valid_loss))
#         print(
#             f"Validation Error (New Trajectories): \n Accuracy: {(100 * new_valid_accuracy):>0.1f}%, Avg loss: {new_valid_loss:>8f} \n")
#         print(f"Validation Error (New Trajectories): \n Precision: {(100 * new_valid_precision):>0.1f}% \n")
#         print(f"Validation Error (New Trajectories): \n Recall: {(100 * new_valid_recall):>0.1f}% \n")
#
#         running_valid_loss, valid_correct = 0.0, 0
#         valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0
#
#         running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn = (
#             _validate_perf(old_test_dataloader, classifier, running_valid_loss,
#                            valid_correct, valid_tp, valid_fp, valid_tn, valid_fn)
#         )
#         valid_loss, valid_accuracy, valid_precision, valid_recall = (
#             _calculate_metrics(running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn,
#                                len(old_test_dataloader),
#                                len(old_test_dataloader.dataset))
#         )
#
#         print('[%d] Validation loss (All Trajectories): %.4f' % (epoch + 1, valid_loss))
#         print(
#             f"Validation Error (All Trajectories): \n Accuracy: {(100 * valid_accuracy):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
#         print(f"Validation Error (All Trajectories): \n Precision: {(100 * valid_precision):>0.1f}% \n")
#         print(f"Validation Error (All Trajectories): \n Recall: {(100 * valid_recall):>0.1f}% \n")
#         epoch += 1
#
#         # if valid_accuracy >= 0.95:
#         # if valid_accuracy >= target_acc:
#         # if valid_accuracy >= 0.99:
#         # if new_valid_accuracy >= target_acc:
#         #     logger.log(f'Early stopping at iter {idx + 1} due to desired accuracy reached')
#         #     break
#
#     # Freeze classifier param
#     for param in classifier.parameters():
#         param.requires_grad_(False)
#
#     classifier.eval()
#     th.backends.cudnn.enabled = False
#
#     # classifier_trainset.add_augment_data(np_data=trajectories_data, indices=train_idx)
#     # classifier_testset.add_augment_data(np_data=trajectories_data, indices=test_idx)
#
#     # if configs.model_cfgs.classifier.save_dir is not None:
#     #     pathlib.Path(configs.model_cfgs.classifier.save_dir).mkdir(parents=True, exist_ok=True)
#     #     th.save(classifier.state_dict(), os.path.join(configs.model_cfgs.classifier.save_dir,
#     #                                                   'Classifier-' + str(epoch) + '.pt'))
#
#     # Add to current Logger
#     return (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
#             (new_valid_precision, valid_precision), (new_valid_recall, valid_recall), total_queries
#             )


@registry.register
class PPOLagLearnedBC(PPOLag):
    """PPOLag with learned cost and budget.
    """

    def _init(self) -> None:
        super()._init()
        utils.set_device_omnisafe(self._cfgs.train_cfgs.device)


    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/EpLearnedCost', window_length=50)
        self._logger.register_key('Metrics/EpLearnedBudget')
        self._logger.register_key('Metrics/LagrangeCostLimit')
        self._logger.register_key('Time/UpdateClassifier')
        self._logger.register_key('Classifier/TrainLoss')
        self._logger.register_key('Classifier/ValidLoss')
        self._logger.register_key('Classifier/ValidAccuracy')
        self._logger.register_key('Classifier/ValidPrecision')
        self._logger.register_key('Classifier/ValidRecall')
        self._logger.register_key('Classifier/NewDataValidLoss')
        self._logger.register_key('Classifier/NewDataValidAccuracy')
        self._logger.register_key('Classifier/NewDataValidPrecision')
        self._logger.register_key('Classifier/NewDataValidRecall')
        self._logger.register_key('Classifier/NumRetrainTrajs')
        self._logger.register_key('Classifier/NumRetrainQueries')


    def _init_env(self) -> None:
        self._env: OnPolicyLearnedBCAdapter = OnPolicyLearnedBCAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
                distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
                self._cfgs.algo_cfgs.steps_per_epoch
                // distributed.world_size()
                // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:

        super()._init_model()

        # Budget Cost Classifier
        pt_env, pt_model_type, pt_batch_size = (
            self._cfgs.model_cfgs.classifier.pt_file.split("/")[-1].split("_")
        )

        pt_batch_size = int(pt_batch_size.split(".pt")[0])

        # TODO: add dropout as kwargs (Note: now only support dropout = 0.0)
        classifier_kwargs = {'feature_dim': self._env.observation_space.shape[0] + self._env.action_space.shape[0],
                             # 'nb_gru_units': pt_hidden_units,
                             # 'batch_size': pt_batch_size,
                             # 'gru_layers': pt_gru_layers,
                             'mlp_arch': self._cfgs.model_cfgs.classifier.decoder_arch}

        self._classifier = classifier_nw_class[pt_model_type](**classifier_kwargs).to(self._device)

        if self._cfgs.model_cfgs.classifier.pt_file is not None:
            self._classifier.load_state_dict(th.load(self._cfgs.model_cfgs.classifier.pt_file, map_location=self._device,
                                                     weights_only=False))

        # Freeze classifier param
        for param in self._classifier.parameters():
            param.requires_grad_(False)

        self._classifier.eval()

        # cudnn does not support backward operations during classifier eval, thus has to be turned off
        th.backends.cudnn.enabled = False

        if self._cfgs.model_cfgs.classifier.lr is not None:
            self.retrain_classifier = True
            self.classifier_optimizer: optim.Optimizer
            self.classifier_optimizer = optim.Adam(
                self._classifier.parameters(),
                lr=self._cfgs.model_cfgs.classifier.lr,
            )

            if self._cfgs.model_cfgs.classifier.train_dataset is None:
                self._classifier_trainset = MujocoNPDataset(self._env_id)
            else:
                self._classifier_trainset = th.load(self._cfgs.model_cfgs.classifier.train_dataset, weights_only=False)

            if self._cfgs.model_cfgs.classifier.test_dataset is None:
                self._classifier_testset = MujocoNPDataset(self._env_id)
            else:
                self._classifier_testset = th.load(self._cfgs.model_cfgs.classifier.test_dataset, weights_only=False)

            # self._classifier_new_trainset = MujocoNPDataset(self._env_id)
            # self._classifier_new_testset = MujocoNPDataset(self._env_id)

            self.max_retrain_epoch = self._cfgs.model_cfgs.classifier.max_retrain_epoch
            self.retrain_target_acc = self._cfgs.model_cfgs.classifier.retrain_target_acc

            self.labeling_noise = self._cfgs.model_cfgs.classifier.labeling_noise
        else:
            self.retrain_classifier = False


    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpLearnedCost')[0]
        assert not np.isnan(Jc), 'learned cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._logger.store({'Metrics/LagrangeCostLimit': self._lagrange.cost_limit})
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        super(PPOLag, self)._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})


    def _update_classifier(self, trajectories: List[pd.DataFrame]) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:

        return _retrain_classifier(
            self._env_id, trajectories, self._classifier, self._classifier_trainset, self._classifier_testset,
            # self._classifier_new_trainset, self._classifier_new_testset,
            self.classifier_optimizer, self._logger,
            self._cfgs, max_epoch=self.max_retrain_epoch, target_acc=self.retrain_target_acc, noise=self.labeling_noise
        )

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        lst_retrain_traj = []
        ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = 0., 0., 0., 0., 0.
        new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = 0., 0., 0., 0.

        num_retrain_trajs, num_retrain_queries = 0, 0
        # num_epochs_last_retrain = 0
        time_update_classifier = 0.0

        for epoch in range(self._cfgs.train_cfgs.epochs):

            # retrain_bool = self.retrain_classifier if (epoch + 1) % 20 == 0 else False

            epoch_time = time.time()

            rollout_time = time.time()
            learned_budget, lst_traj_df = self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
                classifier=self._classifier,
                collect_trajs=True,
            )

            if learned_budget is not None:
                self._lagrange.cost_limit = learned_budget

            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            # num_select_traj = math.ceil(0.6 * len(lst_traj_df))
            num_select_traj = math.ceil(self._cfgs.model_cfgs.classifier.random_retrain_traj * len(lst_traj_df))
            selected_indices = np.random.choice(len(lst_traj_df), num_select_traj, replace=False)
            lst_select_retrain_traj = [lst_traj_df[i] for i in selected_indices]

            lst_retrain_traj += lst_select_retrain_traj

            if (epoch + 1) % 20 == 0:
                update_classifier_time = time.time()
                (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
                 (new_valid_precision, valid_precision), (new_valid_recall, valid_recall), n_queries) = self._update_classifier(lst_retrain_traj)
                time_update_classifier = time.time() - update_classifier_time
                # self._logger.store({'Time/UpdateClassifier': time.time() - update_classifier_time})
                num_retrain_trajs += len(lst_retrain_traj)
                num_retrain_queries += n_queries
                lst_retrain_traj = []

            self._logger.store({'Time/UpdateClassifier': time_update_classifier})

            self._logger.store({'Classifier/TrainLoss': ave_training_loss})
            self._logger.store({'Classifier/ValidLoss': valid_loss})
            self._logger.store({'Classifier/ValidAccuracy': valid_accuracy})
            self._logger.store({'Classifier/ValidPrecision': valid_precision})
            self._logger.store({'Classifier/ValidRecall': valid_recall})

            self._logger.store({'Classifier/NewDataValidLoss': new_valid_loss})
            self._logger.store({'Classifier/NewDataValidAccuracy': new_valid_accuracy})
            self._logger.store({'Classifier/NewDataValidPrecision': new_valid_precision})
            self._logger.store({'Classifier/NewDataValidRecall': new_valid_recall})

            self._logger.store({'Classifier/NumRetrainTrajs': num_retrain_trajs})
            self._logger.store({'Classifier/NumRetrainQueries': num_retrain_queries})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()
                if self._classifier is not None:
                    path = os.path.join(self._logger.log_dir, 'torch_save', f'classifier-{self._logger.current_epoch}.pt')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    th.save(self._classifier.state_dict(), path)

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        ep_learned_cost = self._logger.get_stats('Metrics/EpLearnedCost')[0]
        ep_learned_budget = self._logger.get_stats('Metrics/EpLearnedBudget')[0]

        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len


@registry.register
class PPOLagLearnedH(PPOLag):
    """PPOLag with learned cost and budget.
    """

    def _init(self) -> None:
        self._buf: VectorOnPolicyBufferH = VectorOnPolicyBufferH(
            obs_space=self._env.observation_space,
            hidden_obs_size=self._cfgs.model_cfgs.classifier.hidden_dim,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )
        # self._lagrange: LagrangeH = LagrangeH(**self._cfgs.lagrange_cfgs)
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
        utils.set_device_omnisafe(self._cfgs.train_cfgs.device)


    def _init_log(self) -> None:
        super()._init_log()

        what_to_save = self._logger._what_to_save
        if self._cfgs.algo_cfgs.hidden_obs_normalize:
            hidden_obs_normalizer = self._env.save()['hidden_obs_normalizer']
            what_to_save['hidden_obs_normalizer'] = hidden_obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key('Metrics/EpNegLogScore', window_length=self._cfgs.algo_cfgs.lagrange_window)
        self._logger.register_key('Metrics/EpProbSafe', window_length=self._cfgs.algo_cfgs.lagrange_window)
        self._logger.register_key('Metrics/EpNormNegLogScore', window_length=self._cfgs.algo_cfgs.lagrange_window)
        self._logger.register_key('Metrics/LagrangeCostLimit')
        self._logger.register_key('Time/UpdateClassifier')
        self._logger.register_key('Classifier/TrainLoss')
        self._logger.register_key('Classifier/ValidLoss')
        self._logger.register_key('Classifier/ValidAccuracy')
        self._logger.register_key('Classifier/ValidPrecision')
        self._logger.register_key('Classifier/ValidRecall')
        self._logger.register_key('Classifier/NewDataValidLoss')
        self._logger.register_key('Classifier/NewDataValidAccuracy')
        self._logger.register_key('Classifier/NewDataValidPrecision')
        self._logger.register_key('Classifier/NewDataValidRecall')
        self._logger.register_key('Classifier/NumRetrainTrajs')
        self._logger.register_key('Classifier/NumRetrainQueries')
        self._logger.register_key('Classifier/TrajAbsCV', min_and_max=True)
        self._logger.register_key('Value/AdvC')


    def _init_env(self) -> None:
        self._env: OnPolicyLearnedHAdapter = OnPolicyLearnedHAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
                distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
                self._cfgs.algo_cfgs.steps_per_epoch
                // distributed.world_size()
                // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:

        self._actor_critic: ConstraintActorCriticH = ConstraintActorCriticH(
            obs_space=self._env.observation_space,
            hidden_obs_size=self._cfgs.model_cfgs.classifier.hidden_dim,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

        # GRU Classifier
        pt_model_type = self._cfgs.model_cfgs.classifier.pt_model_type

        # pt_env, pt_model_type, pt_hidden_dim, pt_gru_layer, pt_batch_size = (
        #     self._cfgs.model_cfgs.classifier.pt_file.split("/")[-1].split("_")
        # )

        # pt_batch_size = int(pt_batch_size.split(".pt")[0])
        pt_batch_size = self._cfgs.model_cfgs.classifier.batchsize

        classifier_kwargs = {'feature_dim': self._env.observation_space.shape[0] + self._env.action_space.shape[0],
                             'nb_gru_units': self._cfgs.model_cfgs.classifier.hidden_dim,
                             'batch_size': pt_batch_size,
                             'gru_layers': self._cfgs.model_cfgs.classifier.stack_layer,
                             'dropout': self._cfgs.model_cfgs.classifier.dropout,
                             'mlp_arch': self._cfgs.model_cfgs.classifier.decoder_arch}
        # if isinstance(classifier_nw_class[pt_model_type], DistributionGRU):
        #     classifier_kwargs['loc_offset'] = self._cfgs.model_cfgs.classifier.loc_offset
        #     classifier_kwargs['log_std_offset'] = self._cfgs.model_cfgs.classifier.log_std_offset

        self._classifier = classifier_nw_class[pt_model_type](**classifier_kwargs).to(self._device)

        if self._cfgs.model_cfgs.classifier.pt_file is not None:
            self._classifier.load_state_dict(th.load(self._cfgs.model_cfgs.classifier.pt_file, map_location=self._device,
                                                     weights_only=False))

        # Freeze classifier param
        for param in self._classifier.parameters():
            param.requires_grad_(False)

        self._classifier.eval()

        # cudnn does not support backward operations during classifier eval, thus has to be turned off
        th.backends.cudnn.enabled = False

        if self._cfgs.model_cfgs.classifier.lr is not None:
            self.retrain_classifier = True
            self.classifier_optimizer: optim.Optimizer
            # self.classifier_optimizer = optim.Adam(
            #     self._classifier.parameters(),
            #     lr=self._cfgs.model_cfgs.classifier.lr,
            # )
            self.classifier_optimizer = optim.AdamW(
                self._classifier.parameters(),
                lr=self._cfgs.model_cfgs.classifier.lr,
                weight_decay=1e-4
            )
            th.nn.utils.clip_grad_norm_(self._classifier.parameters(), max_norm=1.0)

            if self._cfgs.model_cfgs.classifier.train_dataset is None:
                self._classifier_trainset = MujocoNPDataset(self._env_id)
            else:
                self._classifier_trainset = th.load(self._cfgs.model_cfgs.classifier.train_dataset, weights_only=False)

            if self._cfgs.model_cfgs.classifier.test_dataset is None:
                self._classifier_testset = MujocoNPDataset(self._env_id)
            else:
                self._classifier_testset = th.load(self._cfgs.model_cfgs.classifier.test_dataset, weights_only=False)

            # self._classifier_new_trainset = MujocoNPDataset(self._env_id)
            # self._classifier_new_testset = MujocoNPDataset(self._env_id)

            self.max_retrain_epoch = self._cfgs.model_cfgs.classifier.max_retrain_epoch
            self.retrain_target_acc = self._cfgs.model_cfgs.classifier.retrain_target_acc

            self.labeling_noise = self._cfgs.model_cfgs.classifier.labeling_noise

        else:
            self.retrain_classifier = False


    def _update(self) -> None:
        # note that logger already uses MPI statistics across all processes..
        # Jc = -math.exp(-self._logger.get_stats('Metrics/EpNegLogScore')[0])
        Jc = -self._logger.get_stats('Metrics/EpProbSafe')[0]
        # print("Jc", Jc)
        # norm_Jc = self._logger.get_stats('Metrics/EpNormNegLogScore')[0]
        assert not np.isnan(Jc), 'learned cost for updating lagrange multiplier is nan'
        # assert not np.isnan(norm_Jc), 'learned cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        # norm_cost_limit = self._env._ep_neglogscore_normalizer.normalize_only(self._lagrange.cost_limit)
        self._logger.store({'Metrics/LagrangeCostLimit': self._lagrange.cost_limit})
        # self._logger.store({'Metrics/LagrangeCostLimit': norm_cost_limit})
        self._lagrange.update_lagrange_multiplier(Jc)
        # self._lagrange.update_lagrange_multiplier(norm_Jc, norm_cost_limit)

        # then update the policy and value function
        data = self._buf.get()
        obs, hidden_obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['hidden_obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        original_hidden_obs = hidden_obs
        old_distribution = self._actor_critic.actor(obs, hidden_obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, hidden_obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                    obs,
                    hidden_obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                # self._update_reward_critic(obs, target_value_r, hidden_obs=hidden_obs)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, hidden_obs=hidden_obs, target_value_c=target_value_c)
                self._update_actor(obs, hidden_obs=hidden_obs, act=act, logp=logp, adv_r=adv_r, adv_c=adv_c)

            new_distribution = self._actor_critic.actor(original_obs, original_hidden_obs)

            kl = (
                th.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Value/AdvC': adv_c.mean().item(),
                'Train/KL': final_kl,
            },
        )

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    # def _update_reward_critic(self, obs: th.Tensor, target_value_r: th.Tensor, hidden_obs: th.Tensor = None) -> None:
    #     self._actor_critic.reward_critic_optimizer.zero_grad()
    #     loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs, hidden_obs)[0], target_value_r)
    #
    #     if self._cfgs.algo_cfgs.use_critic_norm:
    #         for param in self._actor_critic.reward_critic.parameters():
    #             loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
    #
    #     loss.backward()
    #
    #     if self._cfgs.algo_cfgs.use_max_grad_norm:
    #         clip_grad_norm_(
    #             self._actor_critic.reward_critic.parameters(),
    #             self._cfgs.algo_cfgs.max_grad_norm,
    #         )
    #     distributed.avg_grads(self._actor_critic.reward_critic)
    #     self._actor_critic.reward_critic_optimizer.step()
    #
    #     self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: th.Tensor, target_value_c: th.Tensor, hidden_obs: th.Tensor = None) -> None:
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs, hidden_obs)[0], target_value_c)

        # if self._cfgs.algo_cfgs.use_critic_norm:
        if self._cfgs.model_cfgs.cost_critic.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                # loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
                loss += param.pow(2).sum() * self._cfgs.model_cfgs.cost_critic.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: th.Tensor,
        act: th.Tensor,
        logp: th.Tensor,
        adv_r: th.Tensor,
        adv_c: th.Tensor,
        hidden_obs: th.Tensor = None,
    ) -> None:
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, hidden_obs=hidden_obs, act=act, logp=logp, adv=adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _loss_pi(
        self,
        obs: th.Tensor,
        act: th.Tensor,
        logp: th.Tensor,
        adv: th.Tensor,
        hidden_obs: th.Tensor = None,
    ) -> th.Tensor:
        distribution = self._actor_critic.actor(obs, hidden_obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = th.exp(logp_ - logp)
        ratio_cliped = th.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss = -th.min(ratio * adv, ratio_cliped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _update_classifier(self, trajectories: List[pd.DataFrame]) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:

        return _retrain_classifier(
            self._env_id, trajectories, self._classifier, self._classifier_trainset, self._classifier_testset,
            # self._classifier_new_trainset, self._classifier_new_testset,
            self.classifier_optimizer, self._logger,
            self._cfgs, max_epoch=self.max_retrain_epoch, target_acc=self.retrain_target_acc, noise=self.labeling_noise
        )

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        lst_all_traj, lst_all_traj_abs_cv, lst_retrain_traj = [], [], []
        ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = 0., 0., 0., 0., 0.
        new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = 0., 0., 0., 0.

        num_retrain_trajs, num_retrain_queries = 0, 0
        # num_epochs_last_retrain = 0
        time_update_classifier = 0.0
        traj_idx = 0

        for epoch in range(self._cfgs.train_cfgs.epochs):

            # retrain_bool = self.retrain_classifier if (epoch + 1) % 20 == 0 else False
            # retrain_bool = self.retrain_classifier if (epoch + 1) % 25 == 0 else False
            # retrain_bool = self.retrain_classifier if (epoch % 20 == 0) or (epoch == self._cfgs.train_cfgs.epochs - 1) else False

            epoch_time = time.time()

            rollout_time = time.time()
            lst_traj_abs_cv, lst_traj_df = self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
                classifier=self._classifier,
                collect_trajs=True,
            )

            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            for abs_cv in lst_traj_abs_cv:
                self._logger.store({'Classifier/TrajAbsCV': abs_cv})

            if self._cfgs.model_cfgs.classifier.random_retrain_traj is None:
                # # Select top retrain_traj_prop trajectories with the highest CV
                # num_include_traj = max(1, math.ceil(self._cfgs.model_cfgs.classifier.retrain_traj_prop * len(lst_traj_abs_cv)))
                # array_traj_abs_cv = np.array(lst_traj_abs_cv)
                # lst_select_retrain_traj = [lst_traj_df[i] for i in np.argsort(array_traj_abs_cv)[-num_include_traj:]]
                # lst_retrain_traj += lst_select_retrain_traj

                lst_all_traj += lst_traj_df
                lst_all_traj_abs_cv += lst_traj_abs_cv
                num_total_include_traj = math.ceil(self._cfgs.model_cfgs.classifier.retrain_traj_prop * len(lst_all_traj))
                if (num_total_include_traj >= self._cfgs.model_cfgs.classifier.min_retrain_trajs) or (epoch == self._cfgs.train_cfgs.epochs - 1):
                    array_all_traj_abs_cv = np.array(lst_all_traj_abs_cv)
                    lst_retrain_traj = [lst_all_traj[i] for i in np.argsort(array_all_traj_abs_cv)[-num_total_include_traj:]]
                    lst_all_traj, lst_all_traj_abs_cv = [], []

            else:
                # # Randomly Select Trajectories (proportion: random_retrain_traj) for classifier retraining
                # num_select_traj = math.ceil(self._cfgs.model_cfgs.classifier.random_retrain_traj * len(lst_traj_df))
                # selected_indices = np.random.choice(len(lst_traj_df), num_select_traj, replace=False)
                # lst_select_retrain_traj = [lst_traj_df[i] for i in selected_indices]
                # lst_retrain_traj += lst_select_retrain_traj

                lst_all_traj += lst_traj_df
                lst_all_traj_abs_cv += lst_traj_abs_cv
                num_total_include_traj = math.ceil(self._cfgs.model_cfgs.classifier.random_retrain_traj * len(lst_all_traj))
                if (num_total_include_traj >= self._cfgs.model_cfgs.classifier.min_retrain_trajs) or (epoch == self._cfgs.train_cfgs.epochs - 1):
                    selected_indices = np.random.choice(len(lst_all_traj), num_total_include_traj, replace=False)
                    lst_retrain_traj = [lst_all_traj[i] for i in selected_indices]
                    lst_all_traj, lst_all_traj_abs_cv = [], []

            if (len(lst_retrain_traj) >= self._cfgs.model_cfgs.classifier.min_retrain_trajs) or (epoch == self._cfgs.train_cfgs.epochs - 1):
                update_classifier_time = time.time()
                (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
                 (new_valid_precision, valid_precision), (new_valid_recall, valid_recall), n_queries) = self._update_classifier(lst_retrain_traj)
                time_update_classifier = time.time() - update_classifier_time
                # self._logger.store({'Time/UpdateClassifier': time.time() - update_classifier_time})
                num_retrain_trajs += len(lst_retrain_traj)
                num_retrain_queries += n_queries
                lst_retrain_traj = []

            if self._cfgs.logger_cfgs.log_trajs:
                traj_path = os.path.join(self._logger.log_dir, 'trajs')
                os.makedirs(traj_path, exist_ok=True)
                for dataframe in lst_traj_df:
                    traj_filename = os.path.join(traj_path, f'traj-{traj_idx}.csv')
                    dataframe.to_csv(traj_filename, index=False)
                    traj_idx += 1

            #     num_epochs_last_retrain = 0
            # else:
            #     num_epochs_last_retrain += 1
            self._logger.store({'Time/UpdateClassifier': time_update_classifier})

            self._logger.store({'Classifier/TrainLoss': ave_training_loss})
            self._logger.store({'Classifier/ValidLoss': valid_loss})
            self._logger.store({'Classifier/ValidAccuracy': valid_accuracy})
            self._logger.store({'Classifier/ValidPrecision': valid_precision})
            self._logger.store({'Classifier/ValidRecall': valid_recall})

            self._logger.store({'Classifier/NewDataValidLoss': new_valid_loss})
            self._logger.store({'Classifier/NewDataValidAccuracy': new_valid_accuracy})
            self._logger.store({'Classifier/NewDataValidPrecision': new_valid_precision})
            self._logger.store({'Classifier/NewDataValidRecall': new_valid_recall})

            self._logger.store({'Classifier/NumRetrainTrajs': num_retrain_trajs})
            self._logger.store({'Classifier/NumRetrainQueries': num_retrain_queries})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()
                if self._classifier is not None:
                    path = os.path.join(self._logger.log_dir, 'torch_save', f'classifier-{self._logger.current_epoch}.pt')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    th.save(self._classifier.state_dict(), path)

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        ep_neg_logscore = self._logger.get_stats('Metrics/EpNegLogScore')[0]
        ep_norm_neg_logscore = self._logger.get_stats('Metrics/EpNormNegLogScore')[0]

        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len


def get_human_label(frames, epoch) -> np.array:
    fps = 5
    # scale_factor = 3.0  # Double the size, adjust as needed
    #
    # for idx, f_ins in enumerate(frames):
    #     os.makedirs(f'video/epoch{epoch:03d}', exist_ok=True)
    #     writer = imageio.get_writer(f'video/epoch{epoch:03d}/video{idx:03d}.mp4', fps=15)
    #     for frame in f_ins:
    #         # # Get current dimensions
    #         # h, w, c = frame.shape
    #         #
    #         # # Calculate new dimensions
    #         # new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    #         #
    #         # # Resize the frame
    #         # resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    #
    #         writer.append_data(frame)
    #         # writer.append_data(resized_frame)
    #     writer.close()

    height, width, _ = frames[0].shape
    # os.makedirs(f"videos/epoch{epoch:03d}", exist_ok=True)
    os.makedirs(f"videos", exist_ok=True)
    print("Writing to path:", os.path.abspath(f"videos/trajectory_{epoch:03d}.mp4"))
    out = cv2.VideoWriter(f"videos/trajectory_{epoch:03d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        # Convert RGB (from MuJoCo) to BGR (for OpenCV)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Saved trajectory_{epoch}.mp4")

    labels = []
    label_dict = {'0': (0, 'unsafe'), '1': (1, 'safe')}
    s = False
    while not s:
        safe_binary = input(
            f'videos/trajectory_{epoch:03d}.mp4 (0 (unsafe), 1 (safe)): ').strip()
        try:
            label, s = label_dict[safe_binary]
            print(s)
        except:
            s = False
    labels.append(label)
    labels = np.array(labels).reshape(-1, 1)
    return labels


def _retrain_classifier_human(env_id: str, trajectories: List[pd.DataFrame], labels: np.array, classifier: nn.Module,
                              classifier_trainset: HumanNPDataset, classifier_testset: HumanNPDataset,
                              classifier_optimizer: Optimizer, logger: Logger, configs: Config, max_epoch: int,
                              target_acc: float) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:

    assert max_epoch > 0

    trajectories_data = TrajDFData(trajectories, env_id)
    all_idx = np.arange(trajectories_data.get_num_traj())
    np.random.shuffle(all_idx)
    split_idx = math.ceil(len(all_idx) * 0.1)
    train_idx, test_idx = all_idx[split_idx:], all_idx[:split_idx]

    classifier_trainset.add_data(np_data=trajectories_data, labels=labels, indices=all_idx)
    classifier_testset.add_data(np_data=trajectories_data, labels=labels, indices=all_idx)

    new_train_dataset = HumanNPDataset(mujoco_domain=classifier_trainset.domain,
                                       np_data=trajectories_data, labels=labels, indices=all_idx)
                                        # np_data=trajectories_data, indices=train_idx)
    new_train_dataloader = DataLoader(new_train_dataset, batch_size=32, collate_fn=collate_maxlength(150), shuffle=True)
    new_test_dataset = HumanNPDataset(mujoco_domain=classifier_testset.domain,
                                      np_data=trajectories_data, labels=labels, indices=all_idx)
                                       # np_data=trajectories_data, indices=test_idx)
    new_test_dataloader = DataLoader(new_test_dataset, batch_size=32, collate_fn=collate_maxlength(150), shuffle=True)

    total_queries = new_train_dataset.n_queries

    # Unfreeze classifier param
    for param in classifier.parameters():
        param.requires_grad_(True)

    classifier.train()
    th.backends.cudnn.enabled = True

    # Retrain classifier for 10 epochs
    epoch = 0
    ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = float('Inf'), float('Inf'), float('-Inf'), float('-Inf'), float('-Inf')
    new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = float('Inf'), float('Inf'), float('-Inf'), float('-Inf')

    old_train_dataloader = DataLoader(classifier_trainset, batch_size=32,
                                      collate_fn=collate_maxlength(150),
                                      shuffle=True)
    old_train_dataloader_iter = iter(old_train_dataloader)
    old_test_dataloader = DataLoader(classifier_testset, batch_size=32, collate_fn=collate_maxlength(150),
                                     shuffle=True)

    for idx in track(range(max_epoch), description='Updating Classifier...'):
        classifier.train()
        running_loss, running_loss_train = 0.0, 0.0

        for i, data in enumerate(old_train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, input_lengths = data
            labels = labels.reshape(-1, 1)

            loss, num_correct, num_tp, num_fp, num_tn, num_fn = (
                classifier.forward_loss_metrics(inputs, labels.float(), input_lengths, classweight=1.0))

            # zero the parameter gradients
            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_train += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        ave_training_loss = running_loss_train / (len(new_train_dataloader) * 2)

        print('[%d] Training loss: %.4f' % (epoch + 1, ave_training_loss))

        classifier.eval()

        running_valid_loss, valid_correct = 0.0, 0
        valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0

        running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn =(
            _validate_perf(new_test_dataloader, classifier, running_valid_loss,
                           valid_correct, valid_tp, valid_fp, valid_tn, valid_fn)
        )
        new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = (
            _calculate_metrics(running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn,
                               len(new_test_dataloader),
                               len(new_test_dataloader.dataset))
        )

        print('[%d] Validation loss (New Trajectories): %.4f' % (epoch + 1, new_valid_loss))
        print(
            f"Validation Error (New Trajectories): \n Accuracy: {(100 * new_valid_accuracy):>0.1f}%, Avg loss: {new_valid_loss:>8f} \n")
        print(f"Validation Error (New Trajectories): \n Precision: {(100 * new_valid_precision):>0.1f}% \n")
        print(f"Validation Error (New Trajectories): \n Recall: {(100 * new_valid_recall):>0.1f}% \n")

        running_valid_loss, valid_correct = 0.0, 0
        valid_tp, valid_tn, valid_fp, valid_fn = 0, 0, 0, 0

        running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn = (
            _validate_perf(old_test_dataloader, classifier, running_valid_loss,
                           valid_correct, valid_tp, valid_fp, valid_tn, valid_fn)
        )
        valid_loss, valid_accuracy, valid_precision, valid_recall = (
            _calculate_metrics(running_valid_loss, valid_correct, valid_tp, valid_fp, valid_tn, valid_fn,
                               len(old_test_dataloader),
                               len(old_test_dataloader.dataset))
        )

        print('[%d] Validation loss (All Trajectories): %.4f' % (epoch + 1, valid_loss))
        print(
            f"Validation Error (All Trajectories): \n Accuracy: {(100 * valid_accuracy):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
        print(f"Validation Error (All Trajectories): \n Precision: {(100 * valid_precision):>0.1f}% \n")
        print(f"Validation Error (All Trajectories): \n Recall: {(100 * valid_recall):>0.1f}% \n")
        epoch += 1

        # if valid_accuracy >= 0.95:
        # if valid_accuracy >= target_acc:
        # if valid_accuracy >= 0.99:
        # if new_valid_accuracy >= target_acc:
        #     logger.log(f'Early stopping at iter {idx + 1} due to desired accuracy reached')
        #     break

    # Freeze classifier param
    for param in classifier.parameters():
        param.requires_grad_(False)

    classifier.eval()
    th.backends.cudnn.enabled = False

    # classifier_trainset.add_augment_data(np_data=trajectories_data, indices=train_idx)
    # classifier_testset.add_augment_data(np_data=trajectories_data, indices=test_idx)

    # Add to current Logger
    return (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
            (new_valid_precision, valid_precision), (new_valid_recall, valid_recall), total_queries
            )


@registry.register
class PPOLagLearnedHuman(PPOLagLearnedH):

    def _init_env(self) -> None:
        self._env: OnPolicyLearnedHumanAdapter = OnPolicyLearnedHumanAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
                distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
                self._cfgs.algo_cfgs.steps_per_epoch
                // distributed.world_size()
                // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:

        self._actor_critic: ConstraintActorCriticH = ConstraintActorCriticH(
            obs_space=self._env.observation_space,
            hidden_obs_size=self._cfgs.model_cfgs.classifier.hidden_dim,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
            log_std_reduce=2.0,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

        # GRU Classifier
        pt_model_type = self._cfgs.model_cfgs.classifier.pt_model_type

        # pt_env, pt_model_type, pt_hidden_dim, pt_gru_layer, pt_batch_size = (
        #     self._cfgs.model_cfgs.classifier.pt_file.split("/")[-1].split("_")
        # )

        # pt_batch_size = int(pt_batch_size.split(".pt")[0])
        pt_batch_size = self._cfgs.model_cfgs.classifier.batchsize

        classifier_kwargs = {'feature_dim': self._env.observation_space.shape[0] + self._env.action_space.shape[0],
                             'nb_gru_units': self._cfgs.model_cfgs.classifier.hidden_dim,
                             'batch_size': pt_batch_size,
                             'gru_layers': self._cfgs.model_cfgs.classifier.stack_layer,
                             'dropout': self._cfgs.model_cfgs.classifier.dropout,
                             'mlp_arch': self._cfgs.model_cfgs.classifier.decoder_arch}
        # if isinstance(classifier_nw_class[pt_model_type], DistributionGRU):
        #     classifier_kwargs['loc_offset'] = self._cfgs.model_cfgs.classifier.loc_offset
        #     classifier_kwargs['log_std_offset'] = self._cfgs.model_cfgs.classifier.log_std_offset

        self._classifier = classifier_nw_class[pt_model_type](**classifier_kwargs).to(self._device)

        if self._cfgs.model_cfgs.classifier.pt_file is not None:
            self._classifier.load_state_dict(th.load(self._cfgs.model_cfgs.classifier.pt_file, map_location=self._device,
                                                     weights_only=False))

        # Freeze classifier param
        for param in self._classifier.parameters():
            param.requires_grad_(False)

        self._classifier.eval()

        # cudnn does not support backward operations during classifier eval, thus has to be turned off
        th.backends.cudnn.enabled = False

        if self._cfgs.model_cfgs.classifier.lr is not None:
            self.retrain_classifier = True
            self.classifier_optimizer: optim.Optimizer
            # self.classifier_optimizer = optim.Adam(
            #     self._classifier.parameters(),
            #     lr=self._cfgs.model_cfgs.classifier.lr,
            # )
            self.classifier_optimizer = optim.AdamW(
                self._classifier.parameters(),
                lr=self._cfgs.model_cfgs.classifier.lr,
                weight_decay=1e-4
            )
            th.nn.utils.clip_grad_norm_(self._classifier.parameters(), max_norm=1.0)

            if self._cfgs.model_cfgs.classifier.train_dataset is None:
                self._classifier_trainset = HumanNPDataset(self._env_id)
            else:
                self._classifier_trainset = th.load(self._cfgs.model_cfgs.classifier.train_dataset, weights_only=False)

            if self._cfgs.model_cfgs.classifier.test_dataset is None:
                self._classifier_testset = HumanNPDataset(self._env_id)
            else:
                self._classifier_testset = th.load(self._cfgs.model_cfgs.classifier.test_dataset, weights_only=False)

            # self._classifier_new_trainset = MujocoNPDataset(self._env_id)
            # self._classifier_new_testset = MujocoNPDataset(self._env_id)

            self.max_retrain_epoch = self._cfgs.model_cfgs.classifier.max_retrain_epoch
            self.retrain_target_acc = self._cfgs.model_cfgs.classifier.retrain_target_acc
        else:
            self.retrain_classifier = False

    def _update(self, warmup: bool = False) -> None:
        Jc = -self._logger.get_stats('Metrics/EpProbSafe')[0]
        assert not np.isnan(Jc), 'learned cost for updating lagrange multiplier is nan'
        self._logger.store({'Metrics/LagrangeCostLimit': self._lagrange.cost_limit})

        # then update the policy and value function
        data = self._buf.get()
        obs, hidden_obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['hidden_obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        original_hidden_obs = hidden_obs
        old_distribution = self._actor_critic.actor(obs, hidden_obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, hidden_obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                    obs,
                    hidden_obs,
                    act,
                    logp,
                    target_value_r,
                    target_value_c,
                    adv_r,
                    adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                # self._update_reward_critic(obs, target_value_r, hidden_obs=hidden_obs)
                if (self._cfgs.algo_cfgs.use_cost) and (not warmup):
                    self._update_cost_critic(obs, hidden_obs=hidden_obs, target_value_c=target_value_c)
                self._update_actor(obs, hidden_obs=hidden_obs, act=act, logp=logp, adv_r=adv_r, adv_c=adv_c, warmup=warmup)

            new_distribution = self._actor_critic.actor(original_obs, original_hidden_obs)

            kl = (
                th.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Value/AdvC': adv_c.mean().item(),
                'Train/KL': final_kl,
            },
        )

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: th.Tensor,
        act: th.Tensor,
        logp: th.Tensor,
        adv_r: th.Tensor,
        adv_c: th.Tensor,
        hidden_obs: th.Tensor = None,
        warmup: bool = False,
    ) -> None:
        adv = self._compute_adv_surrogate(adv_r, adv_c, warmup=warmup)
        loss = self._loss_pi(obs, hidden_obs=hidden_obs, act=act, logp=logp, adv=adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _compute_adv_surrogate(self, adv_r: th.Tensor, adv_c: th.Tensor, warmup: bool = False) -> th.Tensor:
        if warmup:
            penalty = 0
        else:
            penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
        # return -adv_c

    def _update_classifier(self, trajectories: List[pd.DataFrame], labels: np.array) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:

        return _retrain_classifier_human(
            self._env_id, trajectories, labels, self._classifier, self._classifier_trainset, self._classifier_testset,
            # self._classifier_new_trainset, self._classifier_new_testset,
            self.classifier_optimizer, self._logger,
            self._cfgs, max_epoch=self.max_retrain_epoch, target_acc=self.retrain_target_acc
        )

    def learn(self) -> tuple[float, float, float]:
        start_time = time.time()
        self._logger.log('INFO: Start training')

        lst_all_traj, lst_all_traj_abs_cv, lst_retrain_traj = [], [], []
        ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = 0., 0., 0., 0., 0.
        new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = 0., 0., 0., 0.

        num_retrain_trajs, num_retrain_queries = 0, 0
        # num_epochs_last_retrain = 0
        time_update_classifier = 0.0
        traj_idx = 0

        for epoch in range(self._cfgs.train_cfgs.epochs):

            print("Epoch:", str(epoch))
            if (self._cfgs.train_cfgs.warmup_epochs is None) or (epoch >= self._cfgs.train_cfgs.warmup_epochs):
                warmup = False
            else:
                warmup = True

            epoch_time = time.time()

            rollout_time = time.time()
            lst_traj_abs_cv, lst_traj_df, eval_frames, eval_traj_df = self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
                classifier=self._classifier,
                collect_trajs=True,
                warmup=warmup,
            )

            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update(warmup=warmup)
            self._logger.store({'Time/Update': time.time() - update_time})

            if not warmup:
                # print("frames", eval_frames)
                # print("DF", eval_traj_df)
                labels_np = get_human_label(eval_frames, epoch)

                lst_retrain_traj = [eval_traj_df]

                update_classifier_time = time.time()
                (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
                 (new_valid_precision, valid_precision), (new_valid_recall, valid_recall), n_queries) = self._update_classifier(lst_retrain_traj, labels_np)
                time_update_classifier = time.time() - update_classifier_time
                num_retrain_trajs += len(lst_retrain_traj)
                num_retrain_queries += len(lst_retrain_traj)

            if self._cfgs.logger_cfgs.log_trajs:
                traj_path = os.path.join(self._logger.log_dir, 'trajs')
                os.makedirs(traj_path, exist_ok=True)
                for dataframe in lst_traj_df:
                    traj_filename = os.path.join(traj_path, f'traj-{traj_idx}.csv')
                    dataframe.to_csv(traj_filename, index=False)
                    traj_idx += 1

            #     num_epochs_last_retrain = 0
            # else:
            #     num_epochs_last_retrain += 1
            self._logger.store({'Time/UpdateClassifier': time_update_classifier})

            self._logger.store({'Classifier/TrainLoss': ave_training_loss})
            self._logger.store({'Classifier/ValidLoss': valid_loss})
            self._logger.store({'Classifier/ValidAccuracy': valid_accuracy})
            self._logger.store({'Classifier/ValidPrecision': valid_precision})
            self._logger.store({'Classifier/ValidRecall': valid_recall})

            self._logger.store({'Classifier/NewDataValidLoss': new_valid_loss})
            self._logger.store({'Classifier/NewDataValidAccuracy': new_valid_accuracy})
            self._logger.store({'Classifier/NewDataValidPrecision': new_valid_precision})
            self._logger.store({'Classifier/NewDataValidRecall': new_valid_recall})

            self._logger.store({'Classifier/NumRetrainTrajs': num_retrain_trajs})
            self._logger.store({'Classifier/NumRetrainQueries': num_retrain_queries})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()
                if self._classifier is not None:
                    path = os.path.join(self._logger.log_dir, 'torch_save', f'classifier-{self._logger.current_epoch}.pt')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    th.save(self._classifier.state_dict(), path)

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        ep_neg_logscore = self._logger.get_stats('Metrics/EpNegLogScore')[0]
        ep_norm_neg_logscore = self._logger.get_stats('Metrics/EpNormNegLogScore')[0]

        self._logger.close()

        eval_rewards = []
        for eval_episode in range(10):
            eval_frames, eval_traj_df = self._env.eval_rollout(
                steps_per_episode=self._env.max_episode_steps,
                agent=self._actor_critic,
                classifier=self._classifier,
            )
            # print(eval_traj_df)
            eval_rewards.append(eval_traj_df['r'].sum())

            eval_height, eval_width, _ = eval_frames[0].shape
            os.makedirs(f"videos", exist_ok=True)
            print("Writing to path:", os.path.abspath(f"videos/eval_{eval_episode+1:02d}.mp4"))
            out = cv2.VideoWriter(f"videos/eval_{eval_episode+1:02d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5,
                                  (eval_width, eval_height))

            for eval_frame in eval_frames:
                # Convert RGB (from MuJoCo) to BGR (for OpenCV)
                out.write(cv2.cvtColor(eval_frame, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Saved eval_{eval_episode+1:02d}.mp4")

        print("Eval Rewards Array:", eval_rewards)
        print("Eval Rewards Average:", np.mean(eval_rewards))
        print("Eval Rewards Stdev:", np.std(eval_rewards))
        self._env.close()

        return ep_ret, ep_cost, ep_len


@registry.register
class PPOLagLearnedBCHuman(PPOLagLearnedBC):

    def _init_env(self) -> None:
        self._env: OnPolicyLearnedBCHumanAdapter = OnPolicyLearnedBCHumanAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
                distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
                self._cfgs.algo_cfgs.steps_per_epoch
                // distributed.world_size()
                // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:

        super(PPOLagLearnedBC, self)._init_model()

        pt_model_type = self._cfgs.model_cfgs.classifier.pt_model_type
        pt_batch_size = self._cfgs.model_cfgs.classifier.batchsize

        # TODO: add dropout as kwargs (Note: now only support dropout = 0.0)
        classifier_kwargs = {'feature_dim': self._env.observation_space.shape[0] + self._env.action_space.shape[0],
                             # 'nb_gru_units': pt_hidden_units,
                             # 'batch_size': pt_batch_size,
                             # 'gru_layers': pt_gru_layers,
                             'mlp_arch': self._cfgs.model_cfgs.classifier.decoder_arch}

        self._classifier = classifier_nw_class[pt_model_type](**classifier_kwargs).to(self._device)

        if self._cfgs.model_cfgs.classifier.pt_file is not None:
            self._classifier.load_state_dict(
                th.load(self._cfgs.model_cfgs.classifier.pt_file, map_location=self._device,
                        weights_only=False))

        # Freeze classifier param
        for param in self._classifier.parameters():
            param.requires_grad_(False)

        self._classifier.eval()

        # cudnn does not support backward operations during classifier eval, thus has to be turned off
        th.backends.cudnn.enabled = False

        if self._cfgs.model_cfgs.classifier.lr is not None:
            self.retrain_classifier = True
            self.classifier_optimizer: optim.Optimizer
            self.classifier_optimizer = optim.Adam(
                self._classifier.parameters(),
                lr=self._cfgs.model_cfgs.classifier.lr,
            )

            if self._cfgs.model_cfgs.classifier.train_dataset is None:
                self._classifier_trainset = HumanNPDataset(self._env_id)
            else:
                self._classifier_trainset = th.load(self._cfgs.model_cfgs.classifier.train_dataset, weights_only=False)

            if self._cfgs.model_cfgs.classifier.test_dataset is None:
                self._classifier_testset = HumanNPDataset(self._env_id)
            else:
                self._classifier_testset = th.load(self._cfgs.model_cfgs.classifier.test_dataset, weights_only=False)

            # self._classifier_new_trainset = MujocoNPDataset(self._env_id)
            # self._classifier_new_testset = MujocoNPDataset(self._env_id)

            self.max_retrain_epoch = self._cfgs.model_cfgs.classifier.max_retrain_epoch
            self.retrain_target_acc = self._cfgs.model_cfgs.classifier.retrain_target_acc
        else:
            self.retrain_classifier = False

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpLearnedCost')[0]
        assert not np.isnan(Jc), 'learned cost for updating lagrange multiplier is nan'
        # then update the policy and value function
        super(PPOLag, self)._update()

    def _compute_adv_surrogate(self, adv_r: th.Tensor, adv_c: th.Tensor) -> th.Tensor:
        return -adv_c

    def _update_classifier(self, trajectories: List[pd.DataFrame], labels: np.array = None) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], int]:

        return _retrain_classifier_human(
            self._env_id, trajectories, labels, self._classifier, self._classifier_trainset, self._classifier_testset,
            # self._classifier_new_trainset, self._classifier_new_testset,
            self.classifier_optimizer, self._logger,
            self._cfgs, max_epoch=self.max_retrain_epoch, target_acc=self.retrain_target_acc
        )

    def learn(self) -> tuple[float, float, float]:
        start_time = time.time()
        self._logger.log('INFO: Start training')

        ave_training_loss, valid_loss, valid_accuracy, valid_precision, valid_recall = 0., 0., 0., 0., 0.
        new_valid_loss, new_valid_accuracy, new_valid_precision, new_valid_recall = 0., 0., 0., 0.

        num_retrain_trajs, num_retrain_queries = 0, 0
        # num_epochs_last_retrain = 0
        time_update_classifier = 0.0

        for epoch in range(self._cfgs.train_cfgs.epochs):

            # retrain_bool = self.retrain_classifier if (epoch + 1) % 20 == 0 else False

            epoch_time = time.time()

            rollout_time = time.time()
            learned_budget, lst_traj_df, frames = self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
                classifier=self._classifier,
                collect_trajs=True,
            )

            if learned_budget is not None:
                self._lagrange.cost_limit = learned_budget

            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            labels_np = get_human_label(frames, epoch)

            lst_retrain_traj = lst_traj_df

            update_classifier_time = time.time()
            (ave_training_loss, (new_valid_loss, valid_loss), (new_valid_accuracy, valid_accuracy),
             (new_valid_precision, valid_precision), (new_valid_recall, valid_recall),
             n_queries) = self._update_classifier(lst_retrain_traj, labels_np)
            time_update_classifier = time.time() - update_classifier_time
            # self._logger.store({'Time/UpdateClassifier': time.time() - update_classifier_time})
            num_retrain_trajs += len(lst_retrain_traj)
            num_retrain_queries += n_queries

            self._logger.store({'Time/UpdateClassifier': time_update_classifier})

            self._logger.store({'Classifier/TrainLoss': ave_training_loss})
            self._logger.store({'Classifier/ValidLoss': valid_loss})
            self._logger.store({'Classifier/ValidAccuracy': valid_accuracy})
            self._logger.store({'Classifier/ValidPrecision': valid_precision})
            self._logger.store({'Classifier/ValidRecall': valid_recall})

            self._logger.store({'Classifier/NewDataValidLoss': new_valid_loss})
            self._logger.store({'Classifier/NewDataValidAccuracy': new_valid_accuracy})
            self._logger.store({'Classifier/NewDataValidPrecision': new_valid_precision})
            self._logger.store({'Classifier/NewDataValidRecall': new_valid_recall})

            self._logger.store({'Classifier/NumRetrainTrajs': num_retrain_trajs})
            self._logger.store({'Classifier/NumRetrainQueries': num_retrain_queries})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                    epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()
                if self._classifier is not None:
                    path = os.path.join(self._logger.log_dir, 'torch_save',
                                        f'classifier-{self._logger.current_epoch}.pt')
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    th.save(self._classifier.state_dict(), path)

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        ep_learned_cost = self._logger.get_stats('Metrics/EpLearnedCost')[0]
        ep_learned_budget = self._logger.get_stats('Metrics/EpLearnedBudget')[0]

        self._logger.close()

        for eval_episode in range(10):
            eval_frames = self._env.eval_rollout(
                steps_per_episode=150,
                agent=self._actor_critic,
            )

            eval_height, eval_width, _ = eval_frames[0].shape
            os.makedirs(f"videos", exist_ok=True)
            print("Writing to path:", os.path.abspath(f"videos/eval_{eval_episode+1:02d}.mp4"))
            out = cv2.VideoWriter(f"videos/eval_{eval_episode+1:02d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24,
                                  (eval_width, eval_height))

            for eval_frame in eval_frames:
                # Convert RGB (from MuJoCo) to BGR (for OpenCV)
                out.write(cv2.cvtColor(eval_frame, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Saved eval_{eval_episode+1:02d}.mp4")


        self._env.close()

        return ep_ret, ep_cost, ep_len


