import matplotlib.pyplot as plt
from transformers import TrainerCallback
import numpy as np

class LossLoggingCallback(TrainerCallback):
    """Custom callback to track training and validation losses during training."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.epochs = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs during training."""
        if logs is not None:
            # Log training loss
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                self.epochs.append(state.epoch)
            
            # Log validation loss
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
    
    def get_losses(self):
        """Return the collected loss data."""
        return {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'steps': self.steps,
            'epochs': self.epochs
        }
    
    def reset(self):
        """Reset all collected data."""
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.epochs = []

def plot_losses(loss_callback, save_path='loss_plots.png', show_plot=True):
    """
    Plot training and validation losses from the LossLoggingCallback.
    
    Args:
        loss_callback: LossLoggingCallback instance with collected data
        save_path: Path to save the plot image
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    loss_data = loss_callback.get_losses()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    if loss_data['train_losses']:
        ax1.plot(loss_data['steps'], loss_data['train_losses'], 
                label='Training Loss', color='blue', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add some statistics
        min_loss = min(loss_data['train_losses'])
        final_loss = loss_data['train_losses'][-1]
        ax1.text(0.02, 0.98, f'Min Loss: {min_loss:.4f}\nFinal Loss: {final_loss:.4f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No training loss data available', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Training Loss (No Data)')
    
    # Plot validation loss
    if loss_data['eval_losses']:
        # Create x-axis for eval losses (assuming they occur at epoch boundaries)
        eval_steps = np.linspace(0, max(loss_data['steps']) if loss_data['steps'] else 1, 
                               len(loss_data['eval_losses']))
        
        ax2.plot(eval_steps, loss_data['eval_losses'], 
                label='Validation Loss', color='red', marker='o', linewidth=2, markersize=6)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Training Steps (Approx.)')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add some statistics
        min_eval_loss = min(loss_data['eval_losses'])
        final_eval_loss = loss_data['eval_losses'][-1]
        ax2.text(0.02, 0.98, f'Min Loss: {min_eval_loss:.4f}\nFinal Loss: {final_eval_loss:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No validation loss data available', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Validation Loss (No Data)')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss plot saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    
    return fig

def plot_combined_losses(loss_callback, save_path='combined_loss_plot.png', show_plot=True):
    """
    Plot training and validation losses on the same axes.
    
    Args:
        loss_callback: LossLoggingCallback instance with collected data
        save_path: Path to save the plot image
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    loss_data = loss_callback.get_losses()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot training loss
    if loss_data['train_losses']:
        ax.plot(loss_data['steps'], loss_data['train_losses'], 
               label='Training Loss', color='blue', linewidth=2, alpha=0.8)
    
    # Plot validation loss
    if loss_data['eval_losses']:
        eval_steps = np.linspace(0, max(loss_data['steps']) if loss_data['steps'] else 1, 
                               len(loss_data['eval_losses']))
        ax.plot(eval_steps, loss_data['eval_losses'], 
               label='Validation Loss', color='red', marker='o', linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Combined loss plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig
