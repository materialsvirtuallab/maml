

def serialize_maml_object(instance):
    if instance is None:
        return None
    if hasattr(instance, 'get_config'):
        return {
            'class_name': instance.__class__.__name__,
            'config': instance.get_config()
        }
    if hasattr(instance, '__name__'):
        return instance.__name__
    else:
        raise ValueError('Cannot serialize', instance)


def deserialize_maml_object(identifier, module_objects=None,
                             printable_module_name='object'):
    if isinstance(identifier, dict):
        # dealing with configuration dictionary
        config = identifier
        if 'class_name' not in config or 'config' not in config:
            raise ValueError('Improper config format: ' + str(config))
        class_name = config['class_name']
        module_objects = module_objects or {}
        cls = module_objects.get(class_name)
        if cls is None:
            raise ValueError('Unknown ' + printable_module_name +
                             ': ' + class_name)
        return cls(**config['config'])

    elif isinstance(identifier, str):
        function_name = identifier
        fn = module_objects.get(function_name)
        if fn is None:
            raise ValueError('Unknown ' + printable_module_name +
                             ':' + function_name)
        return fn
