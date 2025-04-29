import tensorflow as tf
from keras import ops

class AdinaW(tf.keras.optimizers.Optimizer): #Careate 29 April 2025 for IJCNN 2025
    """Optimizer that implements the ADINA-W algorithm.
        Adina is compatible with all enhancements proposed for Adam, 
        enabling the application of various improvements originally developed for Adam.

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
            to `1e-7`.
        a, b: Original hyperparameters in ADINA paper.  Defaults to `0.1, 0.9`.

        weight_decay: A float value. Coefficient for weight decay regularization (L2 penalty). 
            Defaults to 0.04.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond". Defaults
            to `False`.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.04,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        a = 0.1,
        b = 0.9,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="AdinaW",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum2"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="velocity"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        wd = ops.cast(self.weight_decay, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(
            ops.cast(self.beta_1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.beta_2, variable.dtype), local_step
        )
        alpha = lr * ops.sqrt(1 - beta_2_power)
        gamma = ops.divide(1.0 - self.a * self.b, self.a)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]


        self.assign_add(
            m, ops.multiply(
                ops.subtract(ops.multiply(gradient, gamma), m), 1 - self.beta_1
                )
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.beta_2
            ),
        )

        if self.weight_decay != 0.0:
          self.assign(
              variable,
              ops.multiply(
                  variable,
                  (1 - ops.multiply(lr, wd)),
              ),
          )

        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(ops.add(ops.divide(m, (1 - beta_1_power)),ops.multiply(gradient, self.b)), alpha),
                ops.add(ops.sqrt(v), self.epsilon)
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "a": self.a,
                "b": self.b,
                "weight_decay": self.weight_decay,
            }
        )
        return config
