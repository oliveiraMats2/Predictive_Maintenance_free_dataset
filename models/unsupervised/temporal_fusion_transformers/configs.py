from pytorch_forecasting.metrics.quantile import QuantileLoss

configs = {"max_prediction_length": 200,
           "max_encoder_length": 30,
           "batch_size": 8,
           "transformation": "softplus",
           "trainer": {
               "max_epochs": 200,
               "accelerator": 'gpu',
               "devices": 1,
               "enable_model_summary": True,
               "gradient_clip_val": 0.1
           },
           "config_model": {
               "learning_rate": 0.0001,
               "hidden_size": 1024,
               "attention_head_size": 4,
               "dropout": 0.1,
               "hidden_continuous_size": 160,
               "output_size": 7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
               "loss": QuantileLoss(),
               "log_interval": 10,
               "reduce_on_plateau_patience": 4
           },

           }
