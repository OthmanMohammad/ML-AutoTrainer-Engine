class DataProcessingRecorder:
    def __init__(self):
        self.steps = []

    def record_step(self, step_name, **params):
        step_record = {'step': step_name, 'parameters': params}
        self.steps.append(step_record)

    def save_steps(self):
        return self.steps

    def load_steps(self, steps):
        self.steps = steps


        