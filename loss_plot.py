import loss_tracker
import utils

def process_and_plot_losses(loss_callback, logger=None):
    """
    Process and plot all loss data from the loss callback.
    
    Args:
        loss_callback: LossLoggingCallback instance with collected data
        logger: Logger instance for output messages (optional)
    
    Returns:
        dict: Summary of loss data and plot paths
    """
    if logger is None:
        # Create a simple print-based logger if none provided
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
        logger = SimpleLogger()
    
    logger.info("Processing and plotting training and validation losses...")
    
    # Get loss data
    loss_data = loss_callback.get_losses()
    
    # Create plots
    plot_paths = {}
    
    try:
        # Generate side-by-side plots
        loss_tracker.plot_losses(loss_callback, save_path='loss_plots.png', show_plot=True)
        plot_paths['separate_plots'] = 'loss_plots.png'
        
        # Generate combined plot
        loss_tracker.plot_combined_losses(loss_callback, save_path='combined_loss_plot.png', show_plot=True)
        plot_paths['combined_plot'] = 'combined_loss_plot.png'
        
        logger.info("Successfully generated loss plots")
        
    except Exception as e:
        logger.warning(f"Error generating plots: {str(e)}")
        plot_paths['error'] = str(e)
    
    # Log summary statistics
    train_count = len(loss_data['train_losses'])
    eval_count = len(loss_data['eval_losses'])
    
    logger.info(f"Collected {train_count} training loss points")
    logger.info(f"Collected {eval_count} validation loss points")
    
    summary = {
        'train_loss_count': train_count,
        'eval_loss_count': eval_count,
        'plot_paths': plot_paths
    }
    
    if loss_data['train_losses']:
        final_train_loss = loss_data['train_losses'][-1]
        min_train_loss = min(loss_data['train_losses'])
        logger.info(f"Final training loss: {final_train_loss:.4f}")
        logger.info(f"Minimum training loss: {min_train_loss:.4f}")
        summary['final_train_loss'] = final_train_loss
        summary['min_train_loss'] = min_train_loss
    
    if loss_data['eval_losses']:
        final_eval_loss = loss_data['eval_losses'][-1]
        min_eval_loss = min(loss_data['eval_losses'])
        logger.info(f"Final validation loss: {final_eval_loss:.4f}")
        logger.info(f"Minimum validation loss: {min_eval_loss:.4f}")
        summary['final_eval_loss'] = final_eval_loss
        summary['min_eval_loss'] = min_eval_loss
    
    # Log training progress analysis
    if train_count > 1:
        loss_improvement = loss_data['train_losses'][0] - loss_data['train_losses'][-1]
        improvement_percent = (loss_improvement / loss_data['train_losses'][0]) * 100
        logger.info(f"Training loss improvement: {loss_improvement:.4f} ({improvement_percent:.2f}%)")
        summary['train_improvement'] = loss_improvement
        summary['train_improvement_percent'] = improvement_percent
    
    if eval_count > 1:
        eval_improvement = loss_data['eval_losses'][0] - loss_data['eval_losses'][-1]
        eval_improvement_percent = (eval_improvement / loss_data['eval_losses'][0]) * 100
        logger.info(f"Validation loss improvement: {eval_improvement:.4f} ({eval_improvement_percent:.2f}%)")
        summary['eval_improvement'] = eval_improvement
        summary['eval_improvement_percent'] = eval_improvement_percent
    
    # Check for potential overfitting
    if eval_count > 1 and train_count > 1:
        if summary.get('final_eval_loss', 0) > summary.get('final_train_loss', 0):
            gap = summary['final_eval_loss'] - summary['final_train_loss']
            logger.info(f"Validation loss is {gap:.4f} higher than training loss - monitor for overfitting")
            summary['overfitting_warning'] = True
            summary['train_eval_gap'] = gap
        else:
            summary['overfitting_warning'] = False
    
    logger.info("Loss processing and plotting completed successfully")
    
    return summary
