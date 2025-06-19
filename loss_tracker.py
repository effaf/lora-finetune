import logging
import matplotlib.pyplot as plt
import os # For saving plot to a file

# Import the actual TrainerCallback from transformers
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

# --- CustomLossLogger Class Definition ---

class CustomLossLogger(TrainerCallback):
    """
    A custom callback class derived from Hugging Face Transformers' TrainerCallback
    to store, log, and plot training and evaluation losses.

    This class intercepts the logging events during training and stores
    the 'loss' (training loss) and 'eval_loss' (evaluation loss) in separate lists.
    It uses a provided logger for messages and matplotlib for plotting the losses.
    """
    def __init__(self, logger: logging.Logger = None):
        super().__init__()
        # Initialize lists to store training and evaluation losses
        self.train_losses = []
        self.eval_losses = []

        self.logger=logger

        self.logger.info("CustomLossLogger initialized.")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        This method is called by the Trainer after each logging step.
        It extracts 'loss' and 'eval_loss' from the logs and stores them.

        Args:
            args: TrainingArguments object containing the training configuration.
            state: TrainerState object containing the current state of training.
            control: TrainerControl object to control the training flow.
            logs: Dictionary containing the logged metrics (e.g., {'loss': 0.5, 'eval_loss': 0.6}).
            **kwargs: Additional keyword arguments.
        """
        if logs is None:
            logs = state.log # Fallback to state.log if logs are not directly passed

        # Check if 'loss' (training loss) is present in the logs
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.logger.info(f"Logged training loss: {logs['loss']:.4f}")

        # Check if 'eval_loss' (evaluation loss) is present in the logs
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
            self.logger.info(f"Logged evaluation loss: {logs['eval_loss']:.4f}")

    def plot_losses(self, output_dir: str = "./", filename: str = "losses_plot.png"):
        """
        Plots the stored training and evaluation losses using matplotlib.
        The plot is saved to a file and also displayed.

        Args:
            output_dir (str): Directory where the plot image will be saved.
            filename (str): Name of the plot image file.
        """
        self.logger.info("--- Plotting Stored Losses ---")

        if not self.train_losses and not self.eval_losses:
            self.logger.info("No losses recorded to plot.")
            return

        plt.figure(figsize=(10, 6))

        if self.train_losses:
            plt.plot(self.train_losses, label='Training Loss', marker='o', linestyle='-')
        else:
            self.logger.warning("No training losses to plot.")

        if self.eval_losses:
            plt.plot(self.eval_losses, label='Evaluation Loss', marker='x', linestyle='--')
        else:
            self.logger.warning("No evaluation losses to plot.")

        plt.title('Training and Evaluation Losses Over Steps')
        plt.xlabel('Log Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)

        try:
            plt.savefig(plot_path)
            self.logger.info(f"Loss plot saved to: {plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plot: {e}")

        plt.show() # Display the plot
        self.logger.info("---------------------\n")
