import importlib

# Dataset imports with graceful error handling
dataset_modules = {
    'hotpotqa': '.hotpotqa',
    'bigcodebench': '.bigcodebench', 
    'pubmed': '.pubmed',
    'amazon': '.amazon',
    'stark': '.stark_prime'
}

# Import each dataset engine with error handling
dataset_engines = {}
for dataset_name, module_path in dataset_modules.items():
    try:
        module = importlib.import_module(module_path, package=__name__)
        dataset_engines[dataset_name] = module.dataset_engine
    except ImportError as e:
        dataset_engines[dataset_name] = None

registered_datasets = dataset_engines
