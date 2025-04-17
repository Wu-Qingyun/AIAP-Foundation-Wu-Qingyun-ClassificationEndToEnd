import os
from src.data_preparation import main as data_prep_main
from src.model_training import main as model_train_main

def main():
    """Main function to run the entire pipeline"""
    print("Starting HDB Resale Flats Classification Pipeline...")
    
    # Step 1: Data Preparation
    print("\nStep 1: Data Preparation")
    print("------------------------")
    data_prep_main()
    
    # Step 2: Model Training and Evaluation
    print("\nStep 2: Model Training and Evaluation")
    print("------------------------------------")
    model_train_main()
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 