import os
import sys
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import LogNormal  # , Normal

import pandas as pd

mujoco_safety_gymnasium_dict = {
    'SafetyAntVelocity-v1': {'state_dim': 27, 'action_dim': 8, 'decoder_arch': [64, 64],},
    'SafetyHalfCheetahVelocity-v1': {'state_dim': 17, 'action_dim': 6, 'decoder_arch': [64, 64],},
    'SafetyHopperVelocity-v1': {'state_dim': 11, 'action_dim': 3, 'decoder_arch': [64, 64],},
    'SafetySwimmerVelocity-v1': {'state_dim': 8, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyWalker2dVelocity-v1': {'state_dim': 17, 'action_dim': 6, 'decoder_arch': [64, 64],},
    'SafetyPointCircle1-v0': {'state_dim': 28, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyPointCircle2-v0': {'state_dim': 28, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyCarCircle1-v0': {'state_dim': 40, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyCarCircle2-v0': {'state_dim': 40, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyAntRun-v0': {'state_dim': 33, 'action_dim': 8, 'decoder_arch': [64, 64],},
    'SafetyBallRun-v0': {'state_dim': 7, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyCarRun-v0': {'state_dim': 7, 'action_dim': 2, 'decoder_arch': [64, 64],},
    'SafetyDroneRun-v0': {'state_dim': 17, 'action_dim': 4, 'decoder_arch': [64, 64],},
}



LOC_MAX = 3
LOC_MIN = -20
LOG_STD_MAX = 2
LOG_STD_MIN = -20
MIN_LOGSCORE = -7

class PtEstGRU(nn.Module):
    def __init__(self, feature_dim=11, nb_gru_units=16, batch_size=256, gru_layers=2, mlp_arch=None, dropout=0.0):
        super().__init__()
        if mlp_arch is None:
            mlp_arch = [64, 64]
        self.hidden = None
        self.feature_dim = feature_dim
        self.nb_gru_units = nb_gru_units
        self.gru_layers = gru_layers
        self.batch_size = batch_size
        self.mlp_arch = mlp_arch
        self.dropout = dropout

        # build actual NN
        self.__build_model()


    def __build_model(self):
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.nb_gru_units,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=self.dropout
        )

        # Decoder Module
        # self.sa_embedding = nn.Identity()
        decoder = []
        # prev_in_features = self.feature_dim + self.nb_gru_units
        prev_in_features = self.nb_gru_units * 2
        for out_features in self.mlp_arch:
            decoder.append(nn.Linear(prev_in_features, out_features))
            decoder.append(nn.ReLU())
            # decoder.append(nn.LayerNorm(out_features))
            prev_in_features = out_features
        self.decoder = nn.Sequential(*decoder)

        self.decoder_output = nn.Linear(self.mlp_arch[-1], 1)
        nn.init.normal_(self.decoder_output.weight, mean=-0.5, std=0.1)
        nn.init.constant_(self.decoder_output.bias, -5.0)

    def init_hidden(self, init_h=None):

        if init_h is None:
            # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
            hidden_h = th.zeros(self.gru_layers, self.batch_size, self.nb_gru_units)
        else:
            hidden_h = init_h

        hidden_h = Variable(hidden_h)

        return hidden_h

    def forward(self, x, x_lengths, init_h=None):
        # reset the hidden state. Must be done before you run a new batch
        self.hidden = self.init_hidden(init_h)
        # print(self.hidden)

        batch_size, seq_len, feature_dim = x.size()
        x_clone = x.clone().swapaxes(0, 1)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the GRU
        x = th.nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu(), enforce_sorted=False, batch_first=True)

        # now run through GRU
        x, self.hidden = self.gru(x, self.hidden)
        x_unpack = th.nn.utils.rnn.pad_packed_sequence(x, batch_first=False, padding_value=0.0)
        # print(torch.exp((-nn.ReLU()(self.out(X_unpack[0]))).sum(dim=0)))
        # print(X_unpack[0][:-1].shape)

        if init_h is None:
            h0 = th.zeros(1, batch_size, self.nb_gru_units)  # self.batch, Not batch_size
        else:
            h0 = init_h[-1:]

        # combinedH = th.cat((h0, x_unpack[0][:-1]), 0)
        h_t_vector = th.cat((h0, x_unpack[0][:-1]), 0)
        h_tplusone_vector = x_unpack[0]

        # sa_embed = self.sa_embedding(x_clone)
        # combinedSAH = th.cat((sa_embed, combinedH), -1)
        combined_two_h = th.cat((h_t_vector, h_tplusone_vector), -1)

        # log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(combinedSAH)
        log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(combined_two_h)

        y_hat = th.exp(log_scores.sum(dim=0))
        # meanH = X_unpack[0].sum(dim=0) / X_unpack[1][:, None].to(device)
        # y_hat = self.class_output(meanH)

        # predicted probability, log C with shape [T, B, 1] (mean and variance)
        return y_hat, {'log_scores': log_scores, 'mean': log_scores_mean, 'var': log_scores_variance}, x_unpack[0], self.hidden

    def forward_loss_metrics(self, x, y, x_lengths):

        # forward + backward + optimize
        outputs, dict_log_c_out, h_out, _ = self(x, x_lengths)
        loss = self.loss(outputs, y.float())

        y = y.bool()
        correct = ((outputs > 0.5) == y).sum().item()
        tp = ((outputs > 0.5) & y).sum().item()
        fp = ((outputs > 0.5) & y.logical_not()).sum().item()
        tn = ((outputs <= 0.5) & y.logical_not()).sum().item()
        fn = ((outputs <= 0.5) & y).sum().item()

        return loss, correct, tp, fp, tn, fn

    def _calc_logscores(self, concat_two_h):
        log_scores = -nn.ReLU()(self.decoder_output(self.decoder(concat_two_h)))
        # return log_scores, mean, variance
        return th.clamp(log_scores, MIN_LOGSCORE, 0), None, None

    @staticmethod
    def loss(y_hat, y, classweight=1.):

        loss = nn.BCELoss()
        loss = loss(y_hat, y)

        return loss

        # bce_loss = nn.BCELoss(reduction='none')
        # interim_loss = bce_loss(y_hat, y)
        # # weights = torch.ones_like(y) + (y == 0).float()
        # class_zero_weight = classweight / (classweight + 1)
        # class_one_weight = 1 - class_zero_weight
        # weights = th.zeros_like(y) + class_zero_weight * (y == 0) + class_one_weight * (y == 1)
        #
        # # print("y")
        # # print(y)
        # # print("weights")
        # # print(weights)
        #
        # return th.mean(2 * weights * interim_loss)


class DistributionGRU(PtEstGRU):
    def __init__(self, feature_dim=11, nb_gru_units=16, batch_size=256, gru_layers=2, mlp_arch=None, dropout=0.0,
                 loc_offset=0.0, log_std_offset=0.0):
        super().__init__(feature_dim, nb_gru_units, batch_size, gru_layers, mlp_arch, dropout)
        # self.decoder_output = nn.Linear(self.mlp_arch[-1], 2)
        self.decoder_output_logstd = nn.Linear(self.mlp_arch[-1], 1)
        self.loc_offset = loc_offset
        self.log_std_offset = log_std_offset

    # def __build_model(self):
    #     print("Dist GRU build model")
    #     super().__build_model()
    #     self.decoder_output = nn.Linear(256, 2)
    #     print("Completed dist GRU build model")

    def _calc_logscores(self, concat_two_h):
        # [T, B, 2]
        loc_params = self.decoder_output(self.decoder(concat_two_h)) - self.loc_offset
        loc_params = th.clamp(loc_params, LOC_MIN, LOC_MAX)

        log_std_params = self.decoder_output_logstd(self.decoder(concat_two_h)) - self.log_std_offset
        log_std_params = th.clamp(log_std_params, LOG_STD_MIN, LOG_STD_MAX)
        score_std = th.ones_like(loc_params) * log_std_params.exp()
        distributions = LogNormal(loc_params, score_std)
        log_scores = -distributions.rsample()  # [T, B]

        # return th.clamp(log_scores.unsqueeze(-1), MIN_LOGSCORE, 0), -distributions.mean, distributions.variance
        return th.clamp(log_scores, MIN_LOGSCORE, 0), -distributions.mean, distributions.variance

if __name__ == '__main__':
    # env_id = 'SafetyAntVelocity-v1'
    env_id = sys.argv[1]
    # base_path = '/SSD2/siowmeng/icml25_results/save_traj/exp-x/PPOLagLearnedH_0-0_SafetyAntVelocity-v1/SafetyAntVelocity-v1---55fb775100c88b77dcb231fcd49be70fc21188a3f53dc001b075b74c83bb94f1/PPOLagLearnedH-{SafetyAntVelocity-v1}/seed-000-2025-02-06-08-14-43'
    base_path = sys.argv[2]
    classifier_path = os.path.join(base_path, 'torch_save', 'classifier-100.pt')
    csv_path = os.path.join(base_path, 'trajs')
    traj_files = [f for f in os.listdir(csv_path) if os.path.isfile(os.path.join(csv_path, f))]

    num_safe, num_unsafe = 0, 0
    safe_csv_path = os.path.join(base_path, 'enriched_trajs', 'safe')
    unsafe_csv_path = os.path.join(base_path, 'enriched_trajs', 'unsafe')

    try:
        classifier_kwargs = {
            'feature_dim': mujoco_safety_gymnasium_dict[env_id]['state_dim'] +
                           mujoco_safety_gymnasium_dict[env_id]['action_dim'],
            'nb_gru_units': 4,
            'batch_size': 128,
            'gru_layers': 2,
            'mlp_arch': mujoco_safety_gymnasium_dict[env_id]['decoder_arch']}
        # if isinstance(classifier_nw_class[pt_model_type], DistributionGRU):
        #     classifier_kwargs['loc_offset'] = self._cfgs.model_cfgs.classifier.loc_offset
        #     classifier_kwargs['log_std_offset'] = self._cfgs.model_cfgs.classifier.log_std_offset

        classifier = DistributionGRU(**classifier_kwargs)
        classifier.load_state_dict(th.load(classifier_path, weights_only=False))
    except FileNotFoundError as error:
        raise FileNotFoundError('The classifier is not found in the save directory.') from error

    all_dfs = []
    for idx, filename in enumerate(traj_files):
        if idx % 100 == 0:
            print("Processing file", idx)
        df = pd.read_csv(os.path.join(csv_path, filename))
        orig_obs = th.tensor(df[[colname for colname in df.columns if colname.startswith('s')]].to_numpy(),
                                dtype=th.float32)
        action = th.tensor(df[[colname for colname in df.columns if colname.startswith('a')]].to_numpy(),
                              dtype=th.float32)

        obs_action = th.concat((orig_obs, action), dim=-1).unsqueeze(dim=0)

        obs_action = th.concat((orig_obs, action), dim=-1).unsqueeze(dim=0)
        full_hidden_obs = th.zeros((2, 1, 4))

        prob_feasible, dict_logscores_t, next_hidden_obs_t, next_full_hidden_obs = classifier(
            obs_action,
            th.FloatTensor([obs_action.shape[1]] * obs_action.shape[0]),
            init_h=full_hidden_obs
        )

        logscores_t, mean_logscores_t, var_logscores_t = (
            dict_logscores_t['log_scores'], dict_logscores_t['mean'], dict_logscores_t['var']
        )

        pred_logscores = mean_logscores_t.squeeze()

        df['new_logscore_mean'] = pred_logscores.detach().numpy()
        df['cum_c'] = df['c'].cumsum()

        if df['c'].sum() > 25:
            os.makedirs(unsafe_csv_path, exist_ok=True)
            df.to_csv(os.path.join(unsafe_csv_path, f'traj-{num_unsafe}.csv'), index=False)
            num_unsafe += 1
        else:
            os.makedirs(safe_csv_path, exist_ok=True)
            df.to_csv(os.path.join(safe_csv_path, f'traj-{num_safe}.csv'), index=False)
            num_safe += 1
