mujoco_markov_safety_dict = {
    'SafetyAntVelocity-v1': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetySwimmerVelocity-v1': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyHalfCheetahVelocity-v1': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyHopperVelocity-v1': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyWalker2dVelocity-v1': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointButton1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointButton2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointCircle1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointCircle2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointGoal1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointGoal2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointPush1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyPointPush2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarButton1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarButton2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarCircle1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarCircle2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarGoal1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarGoal2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarPush1-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyCarPush2-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 20,
    },
    'SafetyAntCircle-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 10,
    },
    'SafetyBallCircle-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 10,
    },
    'SafetyCarCircle-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 10,
    },
    'SafetyDroneCircle-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 10,
    },
    'SafetyAntRun-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 5,
    },
    'SafetyBallRun-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        # 'augment_slice': 5,
        'augment_slice': 5,
    },
    'SafetyCarRun-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 5,
    },
    'SafetyDroneRun-v0': {
        'col_name': 'c',
        'total_threshold': 25,
        'augment_slice': 5,
    },

}


class LabeledNPData:

    def __init__(self, input_sa_data, input_cost_data, domain, horizon, markov_cost=True):

        self.domain = domain
        self.horizon = horizon
        self.sa_data = input_sa_data
        self.cost_data = input_cost_data
        self.markov_cost = markov_cost
        self.safe, _ = self.label_data()

    def label_data(self, start_idx=0, end_idx=None):

        if end_idx is None:
            end_idx = self.horizon

        if self.markov_cost:

            cost_threshold = mujoco_markov_safety_dict[self.domain]['total_threshold']
            # satisfy_np = (self.cost_data[start_idx:end_idx] <= cost_threshold / self.horizon)
            satisfy_np = (self.cost_data[start_idx:end_idx] <= 0.0)

            # if self.cost_data[start_idx:end_idx].sum() <= cost_threshold * (end_idx - start_idx) / self.horizon:
            if self.cost_data[start_idx:end_idx].sum() <= cost_threshold:
                safe = True
            else:
                safe = False

            return safe, satisfy_np

        else:
            # Not implemented for now
            return None
