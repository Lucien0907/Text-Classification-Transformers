# Text-Classification-Transformers
Fine-tuning transformer models for text classification

.
├── configs                 # training configuration jason files.
├── data  ──:── banking77              # Multi-cLass Intent classification dataset
├           ├── drug_review_raw        # Multi-cLass Condiction Prediction based on drug review raw dataset 
├           └── drug_review_straified  # Stratified data splits for drug review 
├── models                             # For customised model modules
├── results                            # Storing model checkpoints and logs
├── finetune.py                        # Main script for finetuing the model 
├── Pipfile                            
├── Pipfile.lock
├── preprocessing.ipynb                # Notebook for preprocessing and stratified sampling of the drug dataset
├── presentation.pdf                   # Slides for the presentation
├── README.md                          
└── utils.py                           # Util function for processing data

## run fine-tuning script
$ python finetune.py -cfg ./configs/train_distilbert.json

## config file
{
    "token_hf":"",
    "dataset_name":"banking77",
    "model_id":"distilbert/distilbert-base-uncased",
    "tokenize_args":{
        "truncation":true
    },
    "train_args":{
        "per_device_train_batch_size":32,
        "per_device_eval_batch_size":8,
        "learning_rate":5e-5,
        "num_train_epochs":1,
        "bf16":false, 
        "torch_compile":false, 
        "optim":"adamw_torch", 
        "logging_strategy":"steps",
        "logging_steps":200,
        "evaluation_strategy":"epoch",
        "save_strategy":"epoch",
        "save_total_limit":10,
        "load_best_model_at_end":true,
        "metric_for_best_model":"f1",
        "report_to":"tensorboard" 
    }
}

LoRA implementation is commented out in finetune.py due to to dependancy issues. This pipfile uses torch-1.12 due to hardware limitation of the local machine, which is not compatible with PEFT package. 
Please update torch version to >1.13 and uncomment the snippets in order for LoRA to work.
To enable LoRA add a "lora_config" field to the configuration file.
{
    "token_hf":"",
    "dataset_name":"banking77",
    "model_id":"bert-base-uncased",
    ......
    ......
    "lora_config":{
        "task_type":"SEQ_CLS",
        "r":1,
        "lora_alpha":1,
        "lora_dropout":0.1
    }
}
