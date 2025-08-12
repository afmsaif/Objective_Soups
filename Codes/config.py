"""
Configuration for Multi-Task Learning with MoDo
"""

class Config:
    """Configuration for training"""
    
    # Data paths
    asr_csv_path = "/media/chenlab2/hdd51/saif/asr/covost_data/welsh/cy_asr_train.csv"
    st_csv_path = "/media/chenlab2/hdd51/saif/asr/covost_data/welsh/cy_en_trans_train.csv"
    asr_tokenizer_path = "/media/chenlab2/hdd51/saif/asr/covost_data/welsh/enhanced_code/welsh_char_model_asr_with_ctc.model"
    st_tokenizer_path = "/media/chenlab2/hdd51/saif/asr/covost_data/welsh/enhanced_code/welsh_char_model_trans_with_ctc.model"
    cpc_data_dirs = [
        "/media/chenlab2/hdd51/saif/asr/covost_data/welsh/clips",
        "/media/chenlab2/hdd51/saif/asr/covost_data/english/clips"
    ]
    
    # Model parameters
    encoder_dim = 512
    num_encoder_layers = 8
    dropout = 0.1
    context_frames = 240
    future_frames = 120
    
    # Training parameters
    batch_size = 16
    learning_rate = 5e-4
    num_epochs = 30
    seed = 42
    save_interval = 5
    
    # MoDo parameters
    modo_gamma = 0.01
    modo_rho = 0.001
    
    # Other parameters
    use_cpc = True
    output_dir = "results_modo"
    analyze_conflicts = True
    
    @classmethod
    def to_args(cls):
        """Convert config to argparse Namespace"""
        import argparse
        args = argparse.Namespace()
        
        for key, value in cls.__dict__.items():
            if not key.startswith('__') and not callable(value):
                setattr(args, key, value)
                
        return args

# Add more configurations here if needed
class LargeModelConfig(Config):
    """Configuration for larger model"""
    encoder_dim = 512
    num_encoder_layers = 8
    batch_size = 16
    
class FastTrainingConfig(Config):
    """Configuration for faster training"""
    num_epochs = 10
    save_interval = 2
    analyze_conflicts = False