import importlib

system_modules = {
    'hotpotqa_system': '.hotpotqa.five_components',
    'bigcodebench_system': '.bigcodebench.four_components',
    'pubmed_system': '.pubmed.three_components_with_model_selection',
    'amazon_system': '.amazon.local_models_for_next_item_selection',
    'stark_prime_system': '.stark.bio_system'
}

# Import each system engine with error handling
registered_systems = {}
for system_name, module_path in system_modules.items():
    try:
        module = importlib.import_module(module_path, package=__name__)
        registered_systems[system_name] = module.system_engine
    except ImportError as e:
        print(f"Error importing {system_name} from {module_path}: {e}")
        registered_systems[system_name] = None