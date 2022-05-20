import tensorflow as tf


class InverseCosineScheduler(tf.keras.optimizers.schedules.CosineDecay):
    def __call__(self, step):
        return self.initial_learning_rate - super().__call__(step)


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.num_replicas = tf.cast(
            tf.distribute.get_strategy().num_replicas_in_sync, tf.float32
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) / 5.0

        return lr

    def get_config(self):
        return {}

    @staticmethod
    def from_config(cfg):
        return TransformerSchedule(384)


class OfficialSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

    def __ceil__(self, step):
        step = tf.cast(step + 1, tf.float32)

        return self.d_model ** (-0.5) * tf.math.minimum(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )


class DelayedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
        delay: int,
        default_value: float = 0.0,
    ):
        self.delay = delay
        self.schedule = schedule
        self.default_value = default_value

    def __call__(self, step):
        if step <= self.delay:
            return self.default_value

        return self.schedule(step - self.delay)

    def get_config(self):
        config = self.schedule.get_config()
        config["delay"] = self.delay
        config["default_value"] = self.default_value
        return config


class LinearWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, final_lr, warmup_steps=8000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        self.final_lr = final_lr

    def __call__(self, step):
        return tf.cond(
            step < self.warmup_steps,
            lambda: tf.cast(step, dtype=tf.float32)
            / tf.cast(tf.math.maximum(1, self.warmup_steps), dtype=tf.float32)
            * self.num_replicas,
            lambda: self.final_lr * self.num_replicas,
        )
